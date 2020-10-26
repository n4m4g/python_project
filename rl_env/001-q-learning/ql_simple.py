#!/usr/bin/python3

import time
from typing import List
import numpy as np
import pandas as pd

# -o---X

N_STATE = 50
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 10000
FRESH_TIME = 0.005

def build_table() -> pd.DataFrame:
    """Build a q_table to store states and corresponding actions

    Returns
    -------
    df : pd.DataFrame
        a dataframe that store states and corresponding actions
    """
    
    df = pd.DataFrame(np.zeros((N_STATE, len(ACTIONS))),
                    columns=ACTIONS)
    return df


def choose_action(state: int, q_table: pd.DataFrame):
    """Use state to choose action in q_table

    Parameters
    ----------
    state : int
        environment state
    q_table : pd.DataFrame
        a dataframe that store states and corresponding actions

    Returns
    -------
    action : str
        action that choose from q_table
    """

    state_actions = q_table.iloc[state,:]
    # 10% random choose action
    if np.random.uniform() > EPSILON or state_actions.all()==0:
        action = np.random.choice(ACTIONS)
    # 90% select action with max value
    else:
        action = state_actions.idxmax()
    return action

def env_feedback(state: int, action: str):
    """ environment give feedback depends on state and action

    Parameters
    ----------
    state : int
        environment state
    action : str
        action that agent choose

    Returns
    -------
    n_state : int
        next environment state
    reward: int
        a scalar that determine how good is the state
    """

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
    
def update_env(state: int, episode: int, step_cnt: int):
    """ update the display of environment

    Parameters
    ----------
    state : int
        environment state
    episode : int
        number of exploration
    step_cnt : int
        steps needed to reach the terminal state
    """

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

def playing(q_table: pd.DataFrame) -> pd.DataFrame:
    """Agent learning and environment interaction

    Parameters
    ----------
    q_table : pd.DataFrame
        a dataframe that store states and corresponding actions
        before learning

    Returns
    -------
    q_table : pd.DataFrame
        a dataframe that store states and corresponding actions
        after learning
    """
    for episode in range(EPISODES):
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

def main():
    q_table = build_table()
    print(q_table)
    q_table_update = playing(q_table)
    print(q_table_update)

if __name__ == "__main__":
    main()
