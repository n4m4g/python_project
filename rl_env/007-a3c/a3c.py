import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import multiprocessing as mp

UPDATE_GLOBAL_ITER = mp.cpu_count()
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200
env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        # actor
        self.fc1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)

        # critic
        self.fc2 = nn.Linear(s_dim, 100)
        self.fc3 = nn.Linear(100, 1)
    
        # layer init
        self.set_init(self.fc1,
                      self.mu,
                      self.sigma,
                      self.fc2,
                      self.fc3
                      )
        self.distribution = torch.distributions.Normal

    def set_init(self, *layers):
        for layer in layers:
            layer.weight.data.normal_(0., 0.1)
            layer.bias.data.fill_(0.)

    def forward(self, x):
        # actor
        a1 = F.relu6(self.fc1(x))
        mu = 2*torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001

        # critic
        c1 = F.relu6(self.fc2(x))
        v = self.fc3(c1)

        return mu, sigma, v

    def choose_action(self, s):
        self.eval()
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1,).data, sigma.view(1,).data)
        a = m.sample().numpy()
        return a

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * torch.log(torch.tensor(2*math.pi, dtype=torch.float32)) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = str(name)
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet = gnet
        self.opt = opt
        self.lnet = Net(N_S, N_A)
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        # -16.2736044 ~ 0
        r_norm = 16.2736044/2
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buf_s, buf_a, buf_r = [], [], []
            ep_r = 0
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(torch.tensor(s[None, :], dtype=torch.float32))
                s_, r, done, _ = self.env.step(a.clip(-2, 2))
                if t+1 == MAX_EP_STEP:
                    done = True
                ep_r += r
                buf_s.append(s)
                buf_a.append(a)
                buf_r.append((r+r_norm)/r_norm)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buf_s, buf_a, buf_r, GAMMA)
                    buf_s, buf_a, buf_r = [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0
    else:
        v_s_ = lnet.forward(torch.tensor(s_[None, :], dtype=torch.float32))[-1].data.numpy()[0,0]

        Gt = torch.zeros_like(torch.tensor(br, dtype=torch.float32))
        for idx in reversed(range(len(br))):
            v_s_ = br[idx] + gamma * v_s_
            Gt[idx] = v_s_

        loss = lnet.loss_func(
                torch.tensor(np.vstack(bs), dtype=torch.float32),
                torch.tensor(np.vstack(ba), dtype=torch.float32),
                Gt[:, None].clone().detach()
                )

        opt.zero_grad()
        loss.backward()

        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp.grad

        opt.step()

        lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = ep_r if global_ep_r.value == 0 else \
                global_ep_r.value*0.99+ep_r*0.01
    res_queue.put(global_ep_r.value)
    print(f"{name}, Ep: {global_ep.value}, Ep_r: {global_ep_r.value:.2f}")

if __name__ == "__main__":
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    n_cpu = mp.cpu_count()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, name) for name in range(n_cpu)]
    [w.start() for w in workers]

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
