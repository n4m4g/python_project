#!/usr/bin/python3

import time
import numpy as np
import pandas as pd

from maze_env import Maze
# -o---X

N_STATE = 50
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPSILON = 10000
FRESH_TIME = 0.005

def build_table():
    # build q table
    return pd.DataFrame(np.zeros((N_STATE, len(ACTIONS))),
                    columns=ACTIONS)

def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    # 10% random choose action
    if np.random.uniform() > EPSILON or state_actions.all()==0:
        action = np.random.choice(ACTIONS)
    # 90% select action with max value
    else:
        action = state_actions.idxmax()
    return action

def env_feedback(state, action):
    if action == 'right':
        # N_STATE-1: target position
        # N_STATE-2: left side of target position
        if state == N_STATE-2:
            n_state = 'terminal'
            reward = 10
        else:
            n_state = state+1
            reward = 1
    else:
        if state == 0:
            n_state = 0
        else:
            n_state = state-1
        reward = -1
    return n_state, reward
    
def update_env(state, episode, step_cnt):
    env = ['-']*(N_STATE-1) + ['T']
    if state == 'terminal':
        log = 'Episode {:02d}: total steps = {}'.format(episode, step_cnt)
        print(' {}'.format(log), end='')
        # time.sleep(0.001)
        print()
    else:
        env[state]='o'
        log = ''.join(env)
        print('\r{}'.format(log), end='')
        # time.sleep(FRESH_TIME)

def playing(q_table):
    for episode in range(MAX_EPSILON):
        state = 0
        step_cnt = 0
        is_terminated = False
        update_env(state, episode, step_cnt)
        while not is_terminated:
            action = choose_action(state, q_table)
            n_state, reward = env_feedback(state, action)
            if n_state != 'terminal':
                q_target = reward + GAMMA*q_table.iloc[n_state,:].max()
            else:
                q_target = reward
                is_terminated = True

            q_est = q_table.loc[state, action]
            q_table.loc[state, action] = q_est + ALPHA*(q_target-q_est)
            state = n_state
            update_env(state, episode, step_cnt)
            step_cnt += 1
    return q_table

class RL(object):
    def __init__(self, action_space, lr=0.01, reward_decay=0.9, e_greedy=0.9):
            self.action_space = action_space
            self.lr = lr
            self.gamma = reward_decay
            self.epsilon = e_greedy
            self.q_table = pd.DataFrame(columns=self.action_space)

    def check_state_exist(self, state):
        if not state in self.q_table.index:
            self.q_table = self.q_table.append(
                    pd.Series(
                        [0]*len(self.action_space),
                        index = self.q_table.columns,
                        name=state
                        )
                    )

    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.rand() < self.epsilon:
            state_actions = self.q_table.loc[state, :]
            action = np.random.choice(state_actions[state_actions==np.max(state_actions)].index)
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, *args):
        pass

class QLearning(RL):
    def __init__(self, action_space, lr=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(action_space, lr, reward_decay, e_greedy)
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_pred = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target-q_pred)

def update(RL, env):
    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        steps = 0
        while True:
            env.render()
            action = RL.choose_action(str(state))
            n_state, reward, done = env.step(action)
            RL.learn(str(state), action, reward, str(n_state))
            state = n_state
            steps += 1
            if done:
                break
        print(f"Episode [{episode+1:03d}/{episodes:03d}]: {steps}")
    print('game over')
    env.destroy()

def main():
    # q_table = build_table()
    # print(q_table)
    # q_table_update = playing(q_table)
    # print(q_table_update)

    env = Maze()
    RL = QLearning(action_space=list(range(env.n_action)))
    update(RL, env)

    env.mainloop()

if __name__ == "__main__":
    main()
