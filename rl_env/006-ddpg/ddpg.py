#!/usr/bin/env python3

import numpy as np
import gym

from agent import DDPG

MAX_EPISODES = 200
MAX_EP_STEPS = 200
RENDER = False
ENV_NAME = 'Pendulum-v0'


def main():
    """
    UnboundLocalError: local variable 'RENDER' referenced before assignment

    If the global variable changed in a function without
    declare with a "global" prefix, then the variable here
    will be treat as a local variable

    For example,
    if "RENDER" is not been declared with global prefix,
    access "RENDER" variable will raise UnboundLocalError
    before assign value to "RENDER"
    """

    global RENDER
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
    # print(f"s_dim: {s_dim}, a_dim: {a_dim}, a_bound: {a_bound}")
    # s_dim: 3, a_dim: 1, a_bound: 2.0

    ddpg = DDPG(s_dim, a_dim, a_bound)

    # var: add noise to action
    var = 3
    for i in range(MAX_EPISODES):
        s = env.reset()
        # s : list
        # s.shape = (3,)
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -a_bound, a_bound)
            s_, r, done, info = env.step(a)

            # s : list
            # a : np.float
            # r : float
            # s_ : list
            ddpg.store_transition(s, a, r/10, s_)

            if ddpg.m_pointer > ddpg.capacity:
                var *= 0.9995
                ddpg.learn()

            s = s_
            ep_reward += r

            if done or (j+1) == MAX_EP_STEPS:
                print(f"Episode: {i:03d}")
                print(f"\tReward: {ep_reward:.3f}, Explore: {var:.2f}")
                if ep_reward > -150:
                    RENDER = True
                break
    env.close()


if __name__ == "__main__":
    main()
