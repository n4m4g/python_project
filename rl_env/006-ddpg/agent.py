import numpy as np
import torch
from torch import nn, optim

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000


class Actor_Net(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, a_dim)
        self.fc2.weight.data.normal_(0, 0.1)
        self.a_bound = a_bound

    def forward(self, s):
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = torch.tanh(s)
        a = torch.mul(s, self.a_bound)
        return a


class Critic_Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic_Net, self).__init__()
        self.fc1_s = nn.Linear(s_dim, 30)
        self.fc1_s.weight.data.normal_(0, 0.1)
        self.fc1_a = nn.Linear(a_dim, 30)
        self.fc1_a.weight.data.normal_(0, 0.1)
        self.relu = nn.ReLU()
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        fc_s = self.fc1_s(s)
        fc_a = self.fc1_a(a)
        s_a = self.relu(fc_s+fc_a)
        action_value = self.out(s_a)
        return action_value


class DDPG:
    def __init__(self, s_dim, a_dim, a_bound):
        self.s_dim = s_dim
        self.a_dim = a_dim
        # s, a, r, s_
        self.memory = np.zeros(shape=(MEMORY_CAPACITY, s_dim*2+a_dim+1),
                               dtype=np.float32)
        self.m_pointer = 0
        self.capacity = MEMORY_CAPACITY

        self.actor_eval = Actor_Net(s_dim, a_dim, a_bound)
        self.actor_target = Actor_Net(s_dim, a_dim, a_bound)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.optim_a = optim.Adam(self.actor_eval.parameters(), lr=LR_A)

        self.critic_eval = Critic_Net(s_dim, a_dim)
        self.critic_target = Critic_Net(s_dim, a_dim)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.optim_c = optim.Adam(self.critic_eval.parameters(), lr=LR_C)

        self.loss_td = nn.MSELoss()

    def learn(self):
        self.soft_target_replacement()
        idxs = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bm = self.memory[idxs]
        bs = torch.tensor(bm[:, :self.s_dim],
                          dtype=torch.float32)
        # bs.shape = (batch_size, s_dim)
        ba = torch.tensor(bm[:, self.s_dim:self.s_dim+self.a_dim],
                          dtype=torch.float32)
        # ba.shape = (batch_size, a_dim)
        br = torch.tensor(bm[:, -self.s_dim-1:-self.s_dim],
                          dtype=torch.float32)
        # br.shape = (batch_size, 1)
        bs_ = torch.tensor(bm[:, -self.s_dim:],
                           dtype=torch.float32)
        # bs_.shape = (batch_size, s_dim)

        """
        update actor_eval
        -----------------
        determine how good is the action
        get action from actor_eval(s)
        get q value from critic_eval(s, a)
        to judge the value of (s, a)
        and to maximum q value => minimum -q value
        -torch.mean(q)
        """
        self.optim_a.zero_grad()
        a = self.actor_eval(bs)
        # a.shape = (batch_size, 1)
        q = self.critic_eval(bs, a)
        # q.shape = (batch_size, 1)
        loss_a = -torch.mean(q)
        loss_a.backward()
        self.optim_a.step()

        """
        update critic_eval
        ------------------
        TD(0) algorithm
        get target v from target critic net
        get eval v from eval critic net
        MSE(target_v, eval_v)
        """
        self.optim_c.zero_grad()
        ba_ = self.actor_target(bs_)
        # ba_.shape = (batch_size, 1)
        target_v = br + GAMMA * self.critic_target(bs_, ba_)
        # target_v.shape = (batch_size, 1)
        est_v = self.critic_eval(bs, ba)
        # est_v.shape = (batch_size, 1)
        loss_c = self.loss_td(target_v, est_v)
        loss_c.backward()
        self.optim_c.step()

    def soft_target_replacement(self):
        """ replace parameters of target net with few parameters of eval net
        TAU: 0.01
        """
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_(1-TAU)')
            eval('self.actor_target.' + x +
                 '.data.add_(TAU*self.actor_eval.' + x + '.data)')

        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_(1-TAU)')
            eval('self.critic_target.' + x +
                 '.data.add_(TAU*self.critic_eval.' + x + '.data)')

    def choose_action(self, s):
        with torch.no_grad():
            # s : list
            # s.shape = (3,)
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            # s.shape = (1, 3)
            # a = self.actor_eval(s)[0].detach()
            a = self.actor_eval(s)[0]
            # a.shape = (1,)
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        idx = self.m_pointer % MEMORY_CAPACITY
        self.memory[idx] = transition
        self.m_pointer += 1
