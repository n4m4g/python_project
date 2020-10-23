#!/usr/bin/python3

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchsnooper
from torch import nn, optim
from torch.nn import functional as F

from maze_env import Maze

def update(RL, env):
    step = 0
    episodes = 200
    for episode in range(episodes):
        state = env.reset()
        while True:
            env.render()
            action = RL.choose_action(state)
            n_state, reward, done = env.step(action)
            RL.store_transition(state, action, reward, n_state)
            # accumulate some experience
            if step > 200 and (step+1)%5==0:
                RL.learn()
            state = n_state
            if done:
                break
            step += 1
    print('game over')
    env.destroy()

class Net(nn.Module):
    def __init__(self, n_state, n_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, n_action)
        self.out.weight.data.normal_(0, 0.1)

    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x

class DQN(object):
    def __init__(self, n_state, n_action, lr, reward_decay, e_greedy, 
            replace_target_iter, m_size, batch_size):

        self.est_net, self.target_net = Net(n_state, n_action), Net(n_state, n_action)
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        self.memory_counter = 0
        self.memory_size = m_size
        self.memory = np.zeros((m_size, n_state*2+2)) # (s, a, r, s_)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.est_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.losses = []
    
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < self.epsilon:
            action_values = self.est_net.forward(state)
            # torch.max(tensor, dim) return max value and idx along dim
            # torch.max(tensor, dim)[1] get argmax
            # since only one state as input
            # [0] will get the idx of max value
            action = torch.max(action_values, dim=1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.est_net.state_dict())
        self.learn_step_counter += 1

        sample_idx = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[sample_idx, :]
        batch_s = torch.FloatTensor(batch_memory[:, :self.n_state])
        batch_a = torch.LongTensor(batch_memory[:, self.n_state:self.n_state+1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, self.n_state+1:self.n_state+2])
        batch_s_ = torch.FloatTensor(batch_memory[:, -self.n_state:])
        
        q_est = self.est_net(batch_s).gather(1, batch_a)
        q_next = self.target_net(batch_s_).detach()
        q_target = batch_r + self.gamma*(q_next.max(dim=1)[0].view(self.batch_size, 1))
        loss = self.criterion(q_target, q_est)

        self.optimizer.zero_grad()
        loss.backward()
        print(loss.item())
        self.losses.append(loss.item())
        self.optimizer.step()

    def plot_cost(self):
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.xlabel('training steps')
        plt.ylabel('loss')
        plt.show()

def main():
    env = Maze()
    RL = DQN(
            env.n_state,
            env.n_action,
            lr=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            m_size=2000,
            batch_size=32
            )
    update(RL, env)
    env.mainloop()
    RL.plot_cost()

if __name__ == "__main__":
    main()
