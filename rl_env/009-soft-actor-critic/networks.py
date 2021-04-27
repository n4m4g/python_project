import os
import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
# import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions,
                 fc1_dim=256, fc2_dim=256,
                 name='critic', ckpt_dir='ckpt/sac'):
        super(CriticNetwork, self).__init__()
        self.ckpt_f = os.path.join(ckpt_dir, name+'_sac')

        self.net = nn.Sequential(
            nn.Linear(input_dims[0]+n_actions, fc1_dim),
            F.relu(),
            nn.Linear(fc1_dim, fc2_dim),
            F.relu(),
            nn.Linear(fc2_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, s, a):
        sa = T.cat([s, a], dim=1)
        q = self.net(sa)
        return q

    def save_ckpt(self):
        T.save(self.state_dict(), self.ckpt_f)

    def load_ckpt(self):
        self.load_state_dict(T.load(self.ckpt_f))


class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions,
                 fc1_dim=256, fc2_dim=256,
                 name='value', ckpt_dir='ckpt/sac'):
        super(ValueNetwork, self).__init__()
        self.ckpt_f = os.path.join(ckpt_dir, name+'_sac')

        self.net = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dim),
            F.relu(),
            nn.Linear(fc1_dim, fc2_dim),
            F.relu(),
            nn.Linear(fc2_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, s):
        v = self.net(s)
        return v

    def save_ckpt(self):
        T.save(self.state_dict(), self.ckpt_f)

    def load_ckpt(self):
        self.load_state_dict(T.load(self.ckpt_f))


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, max_action, n_actions,
                 fc1_dim=256, fc2_dim=256,
                 name='actor', ckpt_dir='ckpt/sac'):
        super(ActorNetwork, self).__init__()
        self.ckpt_f = os.path.join(ckpt_dir, name+'_sac')

        self.net = nn.Sequential(
            nn.Linear(*input_dims, fc1_dim),
            F.relu(),
            nn.Linear(fc1_dim, fc2_dim),
            F.relu()
        )
        self.mu = nn.Linear(fc2_dim, n_actions)
        self.sigma = nn.Linear(fc2_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

        self.max_action = T.Tensor(max_action).to(self.device)
        self.reparam_noise = 1e-6

    def forward(self, s):
        prob = self.net(s)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # constraint the standard deviation not be too broad
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, s, reparameterize=True):
        mu, sigma = self.forward(s)
        probs = Normal(mu, sigma)

        # extra noise for sampling
        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()

        action = T.tanh(actions)*self.max_action

        # Confused about the implementation of logprobs
        logprobs = probs.log_prob(actions)
        logprobs -= T.log(1-action.pow(2)+self.reparam_noise)
        logprobs = logprobs.sum(1, keepdim=True)

        return action, logprobs

    def save_ckpt(self):
        T.save(self.state_dict(), self.ckpt_f)

    def load_ckpt(self):
        self.load_state_dict(T.load(self.ckpt_f))
