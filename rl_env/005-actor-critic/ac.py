#!/usr/bin/env python3

from pprint import pprint
import numpy as np
import torch
import gym

from agent import Actor, Critic


def main():
    np.random.seed(2)
    torch.manual_seed(2)

    # Superparameters
    MAX_EPISODE = 3000

    # renders environment if total episode reward
    # is greater then this threshold
    DISPLAY_REWARD_THRESHOLD = 200
    MAX_EP_STEPS = 1000   # maximum time step in one episode
    RENDER = False  # rendering wastes time
    GAMMA = 0.9     # reward discount in TD error
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.005     # learning rate for critic

    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    pprint(env.__dict__)

    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    print(f"N_F: {N_F}, N_A: {N_A}")
    # N_F: 4, N_A: 2

    actor = Actor(N_F, N_A, LR_A)
    critic = Critic(N_F, LR_C, GAMMA)
    running_reward = 0

    for episode in range(MAX_EPISODE):
        s = env.reset()
        # s : list
        # s.shape = (4,)
        t = 0
        track_r = []

        while True:
            if RENDER:
                env.render()

            # actor choose action
            a = actor.choose_action(s)
            # a : scalar
            s_, r, done, info = env.step(a)
            # s.shape = (4,)
            # r : float

            # done means the pole is down
            # last reward give negative reward
            if done:
                r = -20

            track_r.append(r)

            td_err = critic.learn(s, r, s_)
            # td : torch.tensor
            # td_err.shape = (1, 1)
            actor.learn(s, a, td_err)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if episode == 0:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print(f"Episode: {episode+1}")
                print(f"\treward: {ep_rs_sum}")
                print(f"\trunning_reward: {running_reward:.1f}")
                print(f"\tdone: {done}")
                break


if __name__ == "__main__":
    main()
