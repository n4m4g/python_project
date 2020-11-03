import numpy as np
import gym
import torch
from torch import nn, optim

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'

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
        a = torch.tanh(s) # -1 ~ 1
        a = torch.mul(a, self.a_bound) # -2 ~ 2
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
        self.memory = np.zeros(shape=(MEMORY_CAPACITY, s_dim*2+a_dim+1),
                               dtype=np.float32)
        self.m_pointer = 0

        self.actor_eval = Actor_Net(s_dim, a_dim, a_bound)
        self.actor_target = Actor_Net(s_dim, a_dim, a_bound)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_eval = Critic_Net(s_dim, a_dim)
        self.critic_target = Critic_Net(s_dim, a_dim)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        self.optim_a = optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.optim_c = optim.Adam(self.critic_eval.parameters(), lr=LR_C)

        self.loss_td = nn.MSELoss()

    def learn(self):
        self.soft_target_replacement()
        idxs = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bm = self.memory[idxs]
        bs = torch.tensor(bm[:, :self.s_dim], dtype=torch.float32)
        ba = torch.tensor(bm[:, self.s_dim:self.s_dim+self.a_dim], dtype=torch.float32)
        br = torch.tensor(bm[:, -self.s_dim-1:-self.s_dim], dtype=torch.float32)
        bs_ = torch.tensor(bm[:, -self.s_dim:], dtype=torch.float32)

        # update actor_eval
        a = self.actor_eval(bs)
        q = self.critic_eval(bs, a)
        loss_a = -torch.mean(q)
        self.optim_a.zero_grad()
        loss_a.backward()
        self.optim_a.step()

        # update critic_eval
        target_v = br + GAMMA * self.critic_target(bs_, self.actor_target(bs_))
        est_v = self.critic_eval(bs, ba)
        loss_c = self.loss_td(target_v, est_v)
        self.optim_c.zero_grad()
        loss_c.backward()
        self.optim_c.step()
        
    def soft_target_replacement(self):
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_(1-TAU)')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')

        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_(1-TAU)')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a = self.actor_eval(s)[0].detach()
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        idx = self.m_pointer % MEMORY_CAPACITY
        self.memory[idx] = transition
        self.m_pointer += 1

def main():
    global RENDER
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
    print(f"s_dim: {s_dim}, a_dim: {a_dim}, a_bound: {a_bound}")

    ddpg = DDPG(s_dim, a_dim, a_bound)
    
    var = 3
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -a_bound, a_bound)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r/10, s_)

            if ddpg.m_pointer > MEMORY_CAPACITY:
                var *= 0.9995
                ddpg.learn()

            s = s_
            ep_reward += r

            if (j+1) == MAX_EP_STEPS:
                print(f"Episode: {i:03d}, Reward: {ep_reward:.3f}, Explore: {var:.2f}")
                if ep_reward > -150:
                    RENDER = True

if __name__ == "__main__":
    main()
