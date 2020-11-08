import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
LR = 0.002
BATCH = 32
UPDATE_STEPS = 4
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

class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.fc1_a = nn.Linear(s_dim, 100)
        self.mu = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)
        self.softplus = nn.Softplus()

        # critic
        self.fc1_c = nn.Linear(s_dim, 100)
        self.fc2_c = nn.Linear(100, 1)

        self.relu = nn.ReLU6()
        self.distribution = torch.distributions.Normal

        self.set_init(self.fc1_a,
                      self.mu,
                      self.sigma,
                      self.fc1_c,
                      self.fc2_c)

    def set_init(self, *layers):
        for layer in layers:
            layer.weight.data.normal_(0, 0.1)
            layer.bias.data.fill_(0.)

    def forward(self, s):
        # actor
        a = self.fc1_a(s)
        a = self.relu(a)
        mu = 2*torch.tanh(self.mu(a))
        sigma = self.softplus(self.sigma(a))

        # critic
        s = self.fc1_c(s)
        s = self.relu(s)
        v = self.fc2_c(s)
        return mu, sigma, v

    def choose_action(self, s):
        self.eval()
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1), sigma.view(1))
        a = m.sample()
        logprob = m.log_prob(a)
        return a.numpy(), logprob

    def evaluate(self, s, a):
        self.eval()
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        a_logprob = m.log_prob(a)
        entropy = m.entropy()
        _, _, v = self.forward(s)
        return a_logprob, v.squeeze(), entropy


class PPO:
    def __init__(self, s_dim, a_dim):
        self.policy = ActorCritic(s_dim, a_dim)
        self.policy_old = ActorCritic(s_dim, a_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def update(self, bs, ba, br, logprob):
        self.policy.train()
        Gt = torch.zeros_like(torch.tensor(br, dtype=torch.float32))
        discounted_reward = 0

        for idx in reversed(range(len(br))):
            discounted_reward = br[idx] + GAMMA * discounted_reward
            Gt[idx] = discounted_reward

        Gt = (Gt-Gt.mean()) / (Gt.std()+1e-5)

        old_s = torch.tensor(bs, dtype=torch.float32).unsqueeze(0)
        old_a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        old_log_prob = torch.tensor(logprob, dtype=torch.float32).unsqueeze(0)

        for _ in range(UPDATE_STEPS):
            logprobs, v, entropy = self.policy.evaluate(old_s, old_a)
            ratios = torch.exp(logprobs-old_log_prob)

            advantages = Gt-v.detach()
            surrogate1 = ratios*advantages
            surrogate2 = torch.clamp(ratios, 1-EPSILON, 1+EPSILON)*advantages
            loss = -torch.min(surrogate1, surrogate2) + 0.5*self.criterion(v, Gt)-0.01*entropy
            print(loss)
            print(loss.shape)
            print(torch.min(surrogate1, surrogate2).shape)
            print(self.criterion(v, Gt), self.criterion(v, Gt).shape)
            print(entropy.shape)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


if __name__ == "__main__":
    env = gym.make('Pendulum-v0').unwrapped
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    ppo = PPO(s_dim, a_dim)
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buf_s, buf_a, buf_r, buf_logprob = [], [], [], []
        ep_r = 0
        for t in range(EP_LEN):
            a, logprob = ppo.policy_old.choose_action(torch.tensor(s[None, :], dtype=torch.float32))
            s_, r, done, _ = env.step(a)
            buf_s.append(s)
            buf_a.append(a)
            buf_r.append((r+8)/8)
            buf_logprob.append(logprob)

            s = s_
            ep_r += r

            if (t+1)%BATCH==0 or t==EP_LEN-1:
                ppo.update(buf_s, buf_a, buf_r, buf_logprob)
                # v_s_ = ppo.critic(torch.tensor(s_[None,:], dtype=torch.float32)).data.numpy()[0,0]
                # discounted_r = torch.zeros_like(torch.tensor(buf_r, dtype=torch.float32))
                # for idx in reversed(range(len(buf_r))):
                #     v_s_ = buf_r[idx] + GAMMA * v_s_
                #     discounted_r[idx] = v_s_

                # ppo.update(torch.tensor(np.vstack(buf_s), dtype=torch.float32),
                #            torch.tensor(np.vstack(buf_a), dtype=torch.float32),
                #            discounted_r[:, None].clone().detach())
                buf_s, buf_a, buf_r, buf_logprob = [], [], [], []

        if ep==0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9+ep_r*0.1)

        print(f"Ep: {ep}, Ep_r: {ep_r}")

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()







    
