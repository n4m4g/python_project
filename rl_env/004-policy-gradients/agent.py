import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.Tanh(),
                nn.Linear(32, n_actions))

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.3)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0.1)


class PolicyGradient(object):
    def __init__(self, n_features, n_actions, lr, reward_decay):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = reward_decay

        self.ep_s = []
        self.ep_a = []
        self.ep_r = []

        self.net = Net(n_features, n_actions)
        self.net.apply(init_weights)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # nn.CrossEntropy = nn.LogSoftmax + nn.NLLLoss
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def choose_action(self, s):
        # o.shape = (2,)
        s_tensor = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        # o_tensor.shape = (1, 2)
        prob = nn.Softmax(dim=1)(self.net(s_tensor))
        # prob.shape = (1, 3)
        m = Categorical(prob)
        # action = np.arange(prob.shape[1])[m.sample().numpy()[0]]
        action = m.sample().item()
        # action : scalar
        return action

    def store_transition(self, s, a, r):
        # o : list, shape=(2,)
        # a : scalar
        # r : float
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def learn(self):
        o_tensor = torch.tensor(np.vstack(self.ep_s), dtype=torch.float)
        # o_tensor.shape = (batch_size, 2)
        a_tensor = torch.tensor(np.array(self.ep_a), dtype=torch.long)
        # a_tensor.shape = (batch_size)
        vt = self.discount_and_norm_rewards()
        # vt.shape = (batch_size)
        vt_tensor = torch.tensor(vt, dtype=torch.float)
        # vt_tensor.shape = (batch_size)

        # Can’t call numpy() on Variable that requires grad.
        # Use var.detach().numpy()
        # means some of data are not as torch.tensor type

        self.optimizer.zero_grad()

        pred = self.net(o_tensor)
        # pred.shape = (batch_size, 3)

        neg_log_prob = self.criterion(pred, a_tensor)
        # neg_log_prob.shape = (batch_size)
        loss = torch.mean(neg_log_prob*vt_tensor)

        loss.backward()
        self.optimizer.step()

        self.ep_s, self.ep_a, self.ep_r = [], [], []

        return vt

    def discount_and_norm_rewards(self):
        discounted_ep_r = np.zeros_like(self.ep_r)
        # discounted_ep_r.shape = (batch_size)
        running_add = 0
        for t in reversed(range(len(self.ep_r))):
            running_add = self.ep_r[t] + self.gamma*running_add
            discounted_ep_r[t] = running_add

        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)

        return discounted_ep_r
