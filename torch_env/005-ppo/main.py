import gym
import numpy as np


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 3e-4
    n_games = 300
