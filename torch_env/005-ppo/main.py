import os
import gym
import numpy as np

from agent import Agent

if __name__ == "__main__":
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    lr = 3e-4
    n_games = 500

    # env.action_space.n = 2
    # env.observation_space.shape = (4,)

    agent = Agent(s_dim=env.observation_space.shape[0],
                  a_dim=env.action_space.n,
                  batch_size=batch_size,
                  lr=lr,
                  n_epochs=n_epochs)

    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, value = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            pack = [obs, action, prob, value, reward, done]
            agent.remember(*pack)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save()

        print('episode[{}] score: {:.3f} avg_score: {:.3f}'.format(
              i+1, score, avg_score))
        print('learn iter: {}'.format(learn_iters))
