import numpy as np
import torch
from torch import nn, optim
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

class Critic_Net(nn.Module):
    def __init__(self, s_dim):
        super(Critic_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
    def forward(x, self):
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

    def choose_action(self, s):
        self.eval()
        mu, sigma = self.forward(s)
        m = self.distribution(mu.view(1,).data, sigma.view(1,).data)
        a = m.sample().numpy().clip(-2, 2)
        return a


class PPO:
    def __init__(self):


if __name__ == "__main__":
    env = gym.make('Pendulum-v0').unwrapped
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    ppo = PPO()
    
