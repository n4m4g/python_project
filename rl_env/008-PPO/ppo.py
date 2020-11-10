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
1. memory collection
2. discounted reward
3. advantage
4. udpate actor, ppo clip
5. update critic, MSE(advantage)
"""

class Actor(nn.Module):
    def __init__(self, s_dim):
        super(Actor, self).__init__()
        self.fc1_a = nn.Linear(s_dim, 100)
        self.mu = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU6()
        self.distribution = torch.distributions.Normal
        self.set_init(self.fc1_a,
                      self.mu,
                      self.sigma)

    def set_init(self, *layers):
        for layer in layers:
            layer.weight.data.normal_(0, 0.1)
            layer.bias.data.fill_(0.)

    def forward(self, s):
        a = self.fc1_a(s)
        a = self.relu(a)
        mu = 2*torch.tanh(self.mu(a))
        sigma = self.softplus(self.sigma(a))
        return mu, sigma

    def choose_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            m = self.distribution(mu.view(1), sigma.view(1))
            a = m.sample()
            logprob = m.log_prob(a)
        return a.numpy(), logprob

    def evaluate(self, s, a):
        mu, sigma = self.forward(s)
        m = self.distribution(mu, sigma)
        a_logprob = m.log_prob(a)
        entropy = m.entropy()
        return a_logprob, entropy

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1_c = nn.Linear(s_dim, 100)
        self.fc2_c = nn.Linear(100, 1)
        self.relu = nn.ReLU6()
        self.set_init(self.fc1_c,
                      self.fc2_c)

    def set_init(self, *layers):
        for layer in layers:
            layer.weight.data.normal_(0, 0.1)
            layer.bias.data.fill_(0.)

    def forward(self, s):
        s = self.fc1_c(s)
        s = self.relu(s)
        v = self.fc2_c(s)
        return v

    def evaluate(self, s):
        v = self.forward(s)
        return v

class PPO:
    def __init__(self, s_dim, a_dim):
        self.actor = Actor(s_dim)
        self.actor_old = Actor(s_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.opt_a = optim.Adam(self.actor.parameters(), LR)

        self.critic = Critic(s_dim, a_dim)
        self.opt_c = optim.Adam(self.critic.parameters(), LR)
        self.criterion = nn.MSELoss()

    def update(self, m):
        s = torch.stack(m.s)
        a = torch.stack(m.a)
        logprob = torch.stack(m.logprob)

        r = torch.stack(m.r)
        discounted_reward = torch.zeros_like(r)
        tmp_reward = 0

        for idx in reversed(range(len(r))):
            tmp_reward = r[idx] + GAMMA * tmp_reward
            discounted_reward[idx] = tmp_reward

        discounted_reward = (discounted_reward-discounted_reward.mean()) / (discounted_reward.std()+1e-5)

        # print(s.shape)
        # print(a.shape)
        # print(logprob.shape)
        # print(discounted_reward.shape)
        # assert False

        # sample transition
        # discounted reward
        # advantage
        # update actor
        # update critic
        
        for _ in range(len(m)//BATCH):
            idx = np.random.choice(len(m), BATCH)
            b_s = s[idx]
            b_a = a[idx]
            b_r = discounted_reward[idx]
            b_logprob = logprob[idx]

            logprobs, entropy = self.actor.evaluate(b_s, b_a)
            v = self.critic.evaluate(b_s)
            advantages = b_r-v
            # surr1
            ratios = torch.exp(logprobs-b_logprob)
            surrogate1 = ratios*advantages
            # surr2
            surrogate2 = torch.clamp(ratios, 1-EPSILON, 1+EPSILON)*advantages
            loss_a = -torch.mean(torch.minimum(surrogate1, surrogate2))
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()

            v = self.critic.evaluate(b_s)
            loss_c = self.criterion(b_r, v)
            self.opt_c.zero_grad()
            loss_c.backward()
            self.opt_c.step()

        self.actor_old.load_state_dict(self.actor.state_dict())

class Memory:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.logprob = []

    def save(self, s, a, r, logprob):
        # save as tensor
        self.s.append(torch.Tensor(s))
        self.a.append(torch.Tensor(a))
        self.r.append(torch.Tensor([r]))
        self.logprob.append(logprob)

    def clean(self):
        self.s = []
        self.a = []
        self.r = []
        self.logprob = []

    def __len__(self):
        return len(self.s)

if __name__ == "__main__":
    env = gym.make('Pendulum-v0').unwrapped
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print(f"s_dim: {s_dim}, a_dim: {a_dim}")

    ppo = PPO(s_dim, a_dim)
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        m = Memory()
        ep_r = 0
        step = 0
        for t in range(EP_LEN):
            a, logprob = ppo.actor_old.choose_action(torch.tensor(s[None, :], dtype=torch.float32))
            s_, r, done, _ = env.step(a)
            # print("s, a, s_, r, logprob")
            # # list, list, list, scalor, tensor
            # print(s, a, s_, r, logprob)
            # print(s.shape, a.shape, s_.shape, r.shape, logprob.shape)
            # assert False
            m.save(s, a, (r+8)/8, logprob.detach())

            s = s_
            ep_r += r
            step += 1

            if (t+1)%(BATCH*10)==0 or t==EP_LEN-1:
                ppo.update(m)
                m.clean()

            if done:
                break

        if ep==0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9+ep_r*0.1)

        print(f"Ep: {ep}, Ep_r: {ep_r:.3f}")

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()







    
