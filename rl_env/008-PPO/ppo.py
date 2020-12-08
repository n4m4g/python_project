#!/usr/bin/env python3

'''
sudo apt install swig -y
pip install box2d-py

Morvan
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py

openai
https://spinningup.openai.com/en/latest/algorithms/ppo.html

RL algos getting stuck in local optima
The policy quickly becomes deterministic without sufficiently exploring.
Reduce learning rate significantly or incorporate a method of maintaining
stochasticity. Look up soft actor critic.
'''

import time
import gym
import torch

from agent import Memory, PPO


def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s


def main():
    env_name = 'BipedalWalker-v3'
    render = False
    solved_reward = 60
    log_interval = 20
    max_episodes = 10000
    max_timesteps = 1500

    update_timestep = 8000
    action_std = 0.5
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr = 3e-4
    betas = (0.9, 0.999)

    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # s_dim, a_dim = 24, 4

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # print(env.action_space.high)
    # print(env.action_space.low)
    # assert False

    device = torch.device('cuda')

    memory = Memory()
    ppo = PPO(s_dim, a_dim, action_std, lr, betas,
              gamma, K_epochs, eps_clip, device)

    running_reward = 0
    avg_length = 0
    time_step = 0
    total_time = time.time()

    for eps in range(max_episodes):
        state = env.reset()
        if eps % log_interval == 0:
            start_t = time.time()
        for t in range(max_timesteps):
            time_step += 1
            action = ppo.choose_action(state, memory)
            # action.shape = (a_dim,)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward

            if render:
                env.render()

            if done:
                break

        avg_length += (t+1)

        # solved_reward = 60
        # log_interval = 20
        # max_episodes = 10000
        # max_timesteps = 1500
        # update_timestep = 8000

        if running_reward > log_interval * solved_reward:
            print("### Solved! ###")
            torch.save(ppo.policy.state_dict(), 'ckpt/ppo_solved.pt')
            break

        if (eps+1) % 500 == 0:
            torch.save(ppo.policy.state_dict(), 'ckpt/ppo.pt')

        if (eps+1) % log_interval == 0:
            end_t = time.time()
            m, s = epoch_time(start_t, end_t)
            t_m, t_s = epoch_time(total_time, end_t)

            avg_length = int(avg_length/log_interval)
            running_reward = int(running_reward/log_interval)

            print(f"Episode {eps+1}")
            print(f"\tTime: {m}m {s}s | Total Time: {t_m}m {t_s}s")
            print(f"\tAvg length: {avg_length}")
            print(f"\tAvg reward: {running_reward}")
            running_reward = 0
            avg_length = 0


if __name__ == "__main__":
    main()
