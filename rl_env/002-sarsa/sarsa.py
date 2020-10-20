#!/usr/bin/python3

import numpy as np
import pandas as pd
import time

# -o---X

N_STATE = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPSILON = 15
FRESH_TIME = 0.005

def build_table():
    # build q table
    return pd.DataFrame(np.zeros((N_STATE, len(ACTIONS))),
                    columns=ACTIONS)

def choose_action(state, q_table):
    if state == 'terminal':
        return np.random.choice(ACTIONS)
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
            reward = 1
        else:
            n_state = state+1
            reward = 0
    else:
        if state == 0:
            n_state = 0
        else:
            n_state = state-1
        reward = 0
    return n_state, reward 
    
def update_env(state, episode, step_cnt):
    env = ['-']*(N_STATE-1) + ['T']
    if state == 'terminal':
        log = 'Episode {:02d}: total steps = {}'.format(episode, step_cnt)
        print(' {}'.format(log), end='')
        time.sleep(0.1)
        print()
    else:
        env[state]='o'
        log = ''.join(env)
        print('\r{}'.format(log), end='')
        time.sleep(FRESH_TIME)

def playing(q_table):
    for episode in range(MAX_EPSILON):
        state = 0
        step_cnt = 0
        is_terminated = False
        update_env(state, episode, step_cnt)
        action = choose_action(state, q_table)
        while not is_terminated:
            n_state, reward = env_feedback(state, action)
            n_action = choose_action(n_state, q_table)
            if n_state != 'terminal':
                q_target = reward + GAMMA*q_table.loc[n_state,n_action]
            else:
                q_target = reward
                is_terminated = True

            q_est = q_table.loc[state, action]
            q_table.loc[state, action] = q_est + ALPHA*(q_target-q_est)
            state = n_state
            action = n_action
            update_env(state, episode, step_cnt)
            step_cnt += 1
    return q_table

def main():
    q_table = build_table()
    print(q_table)
    q_table_update = playing(q_table)
    print(q_table_update)

if __name__ == "__main__":
    main()
