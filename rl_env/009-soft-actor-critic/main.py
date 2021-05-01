import pybullet_envs
import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent


if __name__ == "__main__":
    _ = pybullet_envs.__name__
    env = gym.make('InvertedPendulumBulletEnv-v0')
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = Agent(input_dims=input_dims, env=env, n_actions=n_actions)
    n_games = 250
    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []
    load_ckpt = False

    if load_ckpt:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, r, done, info = env.step(action)
            score += r
            agent.remember(obs, obs_, action, r, done)
            if not load_ckpt:
                agent.learn()
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            # if not load_ckpt:
            #     agent.save_models()

        print(f'episode {i}, score {score:.1f}, avg_score {avg_score:.1f}')

    if not load_ckpt:
        x = [i+1 for i in range(n_games)]
        plt.plot(x, avg_score_history, label='reward')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig('result.png')
