#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt

from agent import PolicyGradient


def main():
    RENDER = False
    MAX_EXPLORE = 2000

    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped

    print(f"action_space: {env.action_space}")
    print(f"action_space.n: {env.action_space.n}")
    print(f"observation_space: {env.observation_space}")
    print(f"observation_space.shape: {env.observation_space.shape}")
    print(f"observation_space.high: {env.observation_space.high}")
    print(f"observation_space.low: {env.observation_space.low}")
    # action_space: Discrete(3)
    # action_space.n: 3
    # observation_space:
    #     Box(-1.2000000476837158, 0.6000000238418579, (2,), float32)
    # observation_space.shape: (2,)
    # observation_space.high: [0.6  0.07]
    # observation_space.low: [-1.2  -0.07]

    agent = PolicyGradient(
            n_features=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            lr=0.001,
            reward_decay=0.995)

    episodes = 3000
    total_reward = [0]*episodes
    for episode in range(episodes):
        s = env.reset()
        # s : list
        # s.shape = (2,)
        for i in range(MAX_EXPLORE):
            if RENDER:
                env.render()
            a = agent.choose_action(s)
            # a : scalar
            s_, r, done, _ = env.step(a)
            # s : list, shape=(2,)
            # r : float
            # done : bool
            agent.store_transition(s, a, r)

            if done or (i+1) == MAX_EXPLORE:
                ep_rs_sum = sum(agent.ep_r)
                total_reward[episode] = ep_rs_sum
                print(f"Episode: {episode+1}")
                print(f"\treward: {ep_rs_sum}, done: {done}")
                vt = agent.learn()

                if ep_rs_sum > 500:
                    RENDER = True

                if (episode+1) == 30:
                    plt.plot(vt)
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            s = s_


if __name__ == "__main__":
    main()
