from pprint import pprint
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
import gym

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
        self.f1 = nn.Linear(n_features, 20)
        self.f1.weight.data.normal_(0.0, 0.1)
        self.f1.bias.data.fill_(0.1)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(20, n_actions)
        self.f2.weight.data.normal_(0.0, 0.1)
        self.f2.bias.data.fill_(0.1)

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

        s = self.f1(s)
        s = self.relu(s)
        s = self.f2(s)
        return s

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
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        assert s.shape == torch.zeros((1, self.n_features)).shape
        a = torch.tensor([a], dtype=torch.long)
        td = td.clone().detach() 
        # td = torch.tensor(td, dtype=torch.float)

        # acts_prob = self.net(s)
        # log_prob = torch.log(acts_prob[0, a])
        # exp_v = -torch.mean(log_prob*td)

        neg_log_prob = self.criterion(self.net(s), a)
        loss = torch.mean(neg_log_prob*td)

        self.optimizer.zero_grad()
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
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        out = self.net(s)
        probs = nn.Softmax(dim=1)(out)
        m = Categorical(probs)
        idx = m.sample().numpy()
        a = np.arange(probs.shape[1])[idx[0]]
        return a

class Critic_Net(nn.Module):
    def __init__(self, n_features):
        super(Critic_Net, self).__init__()
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
    """Determine the value function
        
        Value function approximation
        Give td error to actor to optimize choosing the action
    """
    def __init__(self, n_features, lr=0.01, GAMMA=0.9):
        self.n_features = n_features
        self.GAMMA = GAMMA
        self.net = Critic_Net(n_features)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def learn(self, s, r, s_):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        assert s.shape == torch.zeros((1, self.n_features)).shape
        s_ = torch.tensor(s_, dtype=torch.float).unsqueeze(0)
        assert s_.shape == torch.zeros((1, self.n_features)).shape
        r = torch.tensor(r, dtype=torch.float)

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

        v_ = self.net(s_)
        v = self.net(s)
        td_err = r + self.GAMMA * v_ - v 
        loss = torch.square(td_err)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_err

def main():
    np.random.seed(2)
    torch.manual_seed(2)

    # Superparameters
    MAX_EPISODE = 3000
    DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 1000   # maximum time step in one episode
    RENDER = False  # rendering wastes time
    GAMMA = 0.9     # reward discount in TD error
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.01     # learning rate for critic

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    
    pprint(env.__dict__)
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # print(N_F, N_A)

    actor = Actor(N_F, N_A, LR_A)
    critic = Critic(N_F, LR_C, GAMMA)

    for episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []

        while True:
            if RENDER:
                env.render()

            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)

            if done:
                r = -20

            track_r.append(r)

            td_err = critic.learn(s, r, s_)
            actor.learn(s, a, td_err)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum

                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print("episode:", episode, "  reward:", int(running_reward))
                break

if __name__ == "__main__":
    main()



