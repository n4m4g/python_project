#!/usr/bin/env python3

'''
sudo apt install swig -y
pip install box2d-py

Morvan
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
'''

import gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import MultivariateNormal


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim, action_std, device):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(s_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, a_dim),
                nn.Tanh())
        self.critic = nn.Sequential(
                nn.Linear(s_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1))
        self.action_var = torch.full(size=(a_dim,),
                                     fill_value=action_std*action_std,
                                     device=device)
        self.device = device

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # state.shape = (1, s_dim)
        action_mean = self.actor(state)
        # action_mean.shape = (1, a_dim)
        cov_mat = torch.diag(self.action_var).to(self.device)
        # cov_mat.shape = (a_dim, a_dim)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        # action.shape = (1, a_dim)
        action_logprob = dist.log_prob(action)
        # action_logprob.shape = (1,)

        memory.states.append(state.detach().cpu().numpy())
        memory.actions.append(action.detach().cpu().numpy())
        memory.logprobs.append(action_logprob.detach().cpu().numpy())

        return action.detach()

    def evaluate(self, state, action):
        # state.shape = (batch_size, s_dim)
        # action.shape = (batch_size, a_dim)
        action_mean = self.actor(state)
        # action_mean.shape = (batch_size, a_dim)
        action_var = self.action_var.expand_as(action_mean)
        # action_var.shape = (batch_size, a_dim)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprob = dist.log_prob(action)
        # action_logprob.shape = (batch_size,)
        dist_entropy = dist.entropy()
        # dist_entropy.shape = (batch_size,)

        state_value = self.critic(state)

        return action_logprob, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, s_dim, a_dim, action_std, lr,
                 betas, gamma, K_epochs, eps_clip, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(s_dim,
                                  a_dim,
                                  action_std,
                                  device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=lr,
                                    betas=betas)
        self.policy_old = ActorCritic(s_dim,
                                      a_dim,
                                      action_std,
                                      device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.criterion = nn.MSELoss()

    def choose_action(self, state, memory):
        # state : list
        # state.shape = (s_dim,)
        with torch.no_grad():
            state = torch.tensor(state,
                                 dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            # state.shape = (1, s_dim)
            action = self.policy_old.act(state, memory)
            action = action.squeeze(0).cpu().numpy()
            # action : numpy array
            # action.shape = (a_dim,)
        return action

    def update(self, memory):
        rewards = [0]*len(memory.rewards)
        disc_reward = 0
        for idx in reversed(range(len(rewards))):
            if memory.is_terminals[idx]:
                disc_reward = 0
            disc_reward = memory.rewards[idx] + self.gamma * disc_reward
            rewards[idx] = disc_reward

        rewards = torch.tensor(rewards,
                               dtype=torch.float32,
                               device=self.device)
        # rewards.shape = (len(memory.rewards),)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # rewards.shape = (len(memory.rewards),)

        old_states = torch.tensor(np.vstack(memory.states),
                                  dtype=torch.float32,
                                  device=self.device)
        # old_states.shape = (batch_size, s_dim)
        old_actions = torch.tensor(np.vstack(memory.actions),
                                   dtype=torch.float32,
                                   device=self.device)
        # old_actions.shape = (batch_size, a_dim)
        old_logprobs = torch.tensor(np.vstack(memory.logprobs),
                                    dtype=torch.float32,
                                    device=self.device)
        # old_logprobs.shape = (batch_size, 1)

        for _ in range(self.K_epochs):
            result = self.policy.evaluate(old_states, old_actions)
            logprobs, state_values, dist_entropy = result

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clip(ratios,
                               1-self.eps_clip,
                               1+self.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2) +
                    0.5 * self.criterion(state_values, rewards) -
                    0.01 * dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    env_name = 'BipedalWalker-v2'
    render = True
    solved_reward = 300
    log_interval = 20
    max_episodes = 10000
    max_timesteps = 1500

    update_timestep = 4000
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

    device = torch.device('cuda')

    memory = Memory()
    ppo = PPO(s_dim, a_dim, action_std, lr, betas,
              gamma, K_epochs, eps_clip, device)

    running_reward = 0
    avg_length = 0
    time_step = 0

    for eps in range(max_episodes):
        state = env.reset()
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

        avg_length += t

        if running_reward > log_interval * solved_reward:
            print("### Solved! ###")
            torch.save(ppo.policy.state_dict(), 'ppo_solved.pt')
            break

        if (eps+1) % 500 == 0:
            torch.save(ppo.policy.state_dict(), 'ppo.pt')

        if (eps+1) % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int(running_reward/log_interval)

            print(f"Episode {eps+1}")
            print(f"\tAvg length: {avg_length}")
            print(f"\tAvg reward: {running_reward}")
            running_reward = 0
            avg_length = 0


if __name__ == "__main__":
    main()
