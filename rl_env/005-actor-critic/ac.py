import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

class Actor_Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(n_features, 20)
        self.f1.weight.data.normal_(0.0, 0.1)
        self.f1.bias.data.fill_(0.1)

        self.relu = nn.ReLU()

        self.f2 = nn.Linear(20, n_actions)
        self.f2.weight.data.normal_(0.0, 0.1)
        self.f2.bias.data.fill_(0.1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, s):
        s = self.f1(s)
        s = self.relu(s)
        s = self.f2(s)
        acts_prob = self.softmax(s)
        return acts_prob

class Actor():
    def __init__(self, n_features, n_actions, lr=0.001):
        self.net = Actor_Net(n_features, n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def learn(self, s, a, td):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        assert s.shape == torch.zeros((1, n_features)).shape
        a = torch.tensor(a, dtype=torch.long)
        td = torch.tensor(td, dtype=torch.float)

        acts_prob = self.net(s)
        log_prob = torch.log(acts_prob[0, a])
        exp_v = -torch.mean(log_prob*td)

        self.optimizer.zero_grad()
        exp_v.backward()
        self.optimizer.step()

        return -exp_v

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        probs = self.net(s)
        m = Categorical(probs)
        idx = m.sample().numpy()
        a = np.arange(probs.shape[1])[idx]
        return a

class Critic_Net(nn.Module):
    def __init__(self, n_features):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(n_features, 20)
        self.f1.weight.data.normal_(0.0, 0.1)
        self.f1.bias.data.fill_(0.1)

        self.relu = nn.ReLU()

        self.f2 = nn.Linear(20, 1)
        self.f2.weight.data.normal_(0.0, 0.1)
        self.f2.bias.data.fill_(0.1)

    def forward(self, s):
        s = self.f1(s)
        s = self.relu(s)
        s = self.f2(s)
        return s

class Critic():
    def __init__(self, n_features, n_actions, lr=0.01):
        self.net = Critic_Net(n_features)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def learn(self, s, r, s_):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        assert s.shape == torch.zeros((1, n_features)).shape
        s_ = torch.tensor(s_, dtype=torch.float).unsqueeze(0)
        assert s_.shape == torch.zeros((1, n_features)).shape
        r = torch.tensor(r, dtype=torch.float)

        v_ = self.net(s_)
        td_err = r + GAMMA * v_ - self.net(s)
        loss = torch.square(td_err)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_err
