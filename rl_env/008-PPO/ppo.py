import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
EPSILON=0.2

"""
1. Build critic net
    - net
    - loss
    - opt
2. Build target actor net, eval actor net
    - 2 net
    - choose action
    - loss
    - opt
3. Update
    - copy state dict
    - advantage from critic
    - opt step actor
    - opt step critic

"""

class Critic_Net(nn.Module):
    def __init__(self, s_dim):
        super(Critic_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
    def forward(self, x):
        return self.net(x)

class Actor_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 100)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)
        self.softplus = nn.Softplus()

    def forward(self, s):
        a = self.fc1(s)
        a = self.relu(a)
        mu = 2*torch.tanh(self.mu(a))
        sigma = self.softplus(self.sigma(a))

        return mu, sigma



class PPO:
    def __init__(self, s_dim, a_dim):
        self.critic = Critic_Net(s_dim)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=C_LR)
        self.criterion_c = nn.MSELoss()

        self.actor_eval = Actor_Net(s_dim, a_dim)
        self.actor_target = Actor_Net(s_dim, a_dim)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.opt_a = optim.Adam(self.actor_eval.parameters(), lr=A_LR)

        self.distribution = torch.distributions.Normal

    def choose_action(self, s):
        self.actor_eval.eval()
        mu, sigma = self.actor_eval(s)
        m = self.distribution(mu.view(1,).data, sigma.view(1,).data)
        a = m.sample().numpy().clip(-2, 2)
        return a

    def update(self, s, a, r):
        self.actor_eval.train()
        self.actor_target.load_state_dict(self.actor_eval.state_dict())

        adv = r - self.critic(s)
        mu, sigma = self.actor_eval(s)
        m_eval = self.distribution(mu, sigma)
        mu, sigma = self.actor_target(s)
        m_target = self.distribution(mu, sigma)
        ratio = m_eval.log_prob(a) / (m_target.log_prob(a)+1e-5)
        surr = ratio * adv
        loss_a = -torch.mean(torch.minimum(surr, torch.clip(ratio, 1-EPSILON, 1+EPSILON)*adv))
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        loss_c = self.criterion_c(r, self.critic(s))
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

if __name__ == "__main__":
    env = gym.make('Pendulum-v0').unwrapped
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    ppo = PPO(s_dim, a_dim)
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buf_s, buf_a, buf_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):
            a = ppo.choose_action(torch.tensor(s[None, :], dtype=torch.float32))
            s_, r, done, _ = env.step(a)
            buf_s.append(s)
            buf_a.append(a)
            buf_r.append((r+8)/8)

            s = s_
            ep_r += r

            if (t+1)%BATCH==0 or t==EP_LEN-1:
                v_s_ = ppo.critic(torch.tensor(s_[None,:], dtype=torch.float32)).data.numpy()[0,0]
                discounted_r = torch.zeros_like(torch.tensor(buf_r, dtype=torch.float32))
                for idx in reversed(range(len(buf_r))):
                    v_s_ = buf_r[idx] + GAMMA * v_s_
                    discounted_r[idx] = v_s_

                ppo.update(torch.tensor(np.vstack(buf_s), dtype=torch.float32),
                           torch.tensor(np.vstack(buf_a), dtype=torch.float32),
                           discounted_r[:, None].clone().detach())
                buf_s, buf_a, buf_r = [], [], []

        if ep==0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9+ep_r*0.1)

        print(f"Ep: {ep}, Ep_r: {ep_r}")

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()







    
