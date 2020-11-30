import torch
from torch import nn, optim
from torch.distributions import Categorical


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0., 0.1)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0.1)


class Actor_Net(nn.Module):
    """action neural network

    Attributes
    ----------
    n_features : int
        number of input feature
    n_actions : int
        number of output feature

    Methods
    -------
    forward(self, s)
        get network output
    """
    def __init__(self, n_features, n_actions):
        super(Actor_Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(n_features, 20),
                nn.ReLU(),
                nn.Linear(20, n_actions))

    def forward(self, s):
        """get network output

        Parameters
        ----------
        s : torch.tensor
            network input (observation)

        Returns
        -------
        s : torch.tensor
            network output (action)
        """
        return self.net(s)


class Actor():
    """Agent that generate action from actor network

        Using policy gradient

    Attributes
    ----------
    n_features : int
        number of input feature
    n_actions : int
        number of output feature
    lr : float
        learning rate

    Methods
    -------
    learn(self, s, a, td)
        update actor network
    choose_action(self, s)
        choose action from actor network
    """

    def __init__(self, n_features, n_actions, lr=0.001):
        self.n_features = n_features
        self.net = Actor_Net(n_features, n_actions)
        self.net.apply(init_weights)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def learn(self, s, a, td):
        """optimize actor network

        Parameters
        ----------
        s : list[float]
            environment state
        a : int
            action
        td : float
            critic give td error to help actor learning
        """
        # s.shape = (4,)
        # a : scalar
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        # s.shape = (1, 4)
        a = torch.tensor([a], dtype=torch.long)
        # a.shape = (1,)
        td = td.clone().detach()
        # td = torch.tensor(td, dtype=torch.float)

        # acts_prob = self.net(s)
        # log_prob = torch.log(acts_prob[0, a])
        # exp_v = -torch.mean(log_prob*td)

        self.optimizer.zero_grad()
        out = self.net(s)
        # out.shape = (1, 2)
        neg_log_prob = self.criterion(out, a)
        # neg_log_prob.shape = (1,)
        loss = torch.mean(neg_log_prob*td)

        loss.backward()
        self.optimizer.step()

    def choose_action(self, s):
        """choose action from actor network

        Parameters
        ----------
        s : list[float]
            environment state

        Returns
        -------
        a : int
            action
        """
        # s.shape = (4,)
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        # s.shape = (1, 4)
        out = self.net(s)
        # out.shape = (1, 2)
        probs = nn.Softmax(dim=1)(out)
        # probs.shape = (1, 2)
        m = Categorical(probs)
        action = m.sample().item()
        # action : scalar
        return action


class Critic_Net(nn.Module):
    def __init__(self, n_features):
        super(Critic_Net, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(n_features, 20),
                nn.ReLU(),
                nn.Linear(20, 1))

    def forward(self, s):
        return self.net(s)


class Critic():
    """Determine the value function

        Value function approximation
        Give td error to actor to optimize choosing the action
    """
    def __init__(self, n_features, lr=0.01, GAMMA=0.9):
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.net = Critic_Net(n_features)
        self.net.apply(init_weights)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def learn(self, s, r, s_):
        # s : list
        # s.shape = (4,)
        # r : float
        # s_ : list
        # s_.shape = (4,)
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        # s.shape = (1, 4)
        s_ = torch.tensor(s_, dtype=torch.float).unsqueeze(0)
        # s_.shape = (1, 4)
        r = torch.tensor(r, dtype=torch.float)
        # r.shape = ([])

        """
        Temporal Difference (TD)

        TD(1)
        -----
        Gt = r(t+1) + gamma*r(t+2) + ...gamma**(T-1)*r(t+T)
        loss = Gt - v(St)

        TD(0)
        ----
        Instead of using the accumulated sum of discounted rewards (Gt)
        we will only look at the immediate reward (Rt+1),
        plus the discount of the estimated value of only 1 step ahead (V(St+1))

        r(t+1) + gamma * v(St+1)
        loss = r(t+1) + gamma * v(St+1) - v(St)
        """

        self.optimizer.zero_grad()
        v_ = self.net(s_)
        # v_.shape = (1, 1)
        v = self.net(s)
        # v.shape = (1, 1)
        td_err = (r + self.GAMMA * v_) - v
        # td_err.shape = (1, 1)
        loss = torch.square(td_err)

        loss.backward()
        self.optimizer.step()

        # td_err.shape = (1, 1)
        return td_err
