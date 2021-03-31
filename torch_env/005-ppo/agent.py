import os
import numpy as np
import torch as T
from torch import nn, optim
from torch.distributions.categorical import Categorical
# from torchsnooper import snoop


class PPOMemory:
    def __init__(self, batch_size):
        # prepare memory buffer
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batch = [indices[i:i+self.batch_size] for i in batch_start]

        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batch)

    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, lr, ckpt_dir='ckpt/'):
        super(Actor, self).__init__()
        self.ckpt_file = os.path.join(ckpt_dir, 'actor.pth')
        self.net = nn.Sequential(
                nn.Linear(s_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, a_dim),
                nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.net(state)
        dist = Categorical(dist)

        return dist

    def save(self):
        T.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(T.load(self.ckpt_file))


class Critic(nn.Module):
    def __init__(self, s_dim, lr, ckpt_dir='ckpt/'):
        super(Critic, self).__init__()
        self.ckpt_file = os.path.join(ckpt_dir, 'critic.pth')
        self.net = nn.Sequential(
                nn.Linear(s_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.net(state)

        return value

    def save(self):
        T.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(T.load(self.ckpt_file))


class Agent:
    def __init__(self, s_dim, a_dim, gamma=0.99, lr=3e-4, gae_lambda=0.95,
                 clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.clip = clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(s_dim, a_dim, lr)
        self.critic = Critic(s_dim, lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store(state, action, prob, val, reward, done)

    def save(self):
        print('... saving models ...')
        self.actor.save()
        self.critic.save()

    def load(self):
        print('... loading models ...')
        self.actor.load()
        self.critic.load()

    def choose_action(self, obs):
        state = T.Tensor([obs]).to(self.actor.device)

        dist = self.actor(state)
        action = dist.sample()
        value = self.critic(state)

        prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value

    # @snoop()
    def learn(self):
        for _ in range(self.n_epochs):
            b_states, b_actions, b_old_probs, b_values, \
                    b_rewards, b_dones, b_batch = \
                    self.memory.generate_batch()
            advantage = np.zeros(len(b_rewards), dtype=np.float32)

            for t in range(len(b_rewards)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(b_rewards)-1):
                    delta = b_rewards[k] + \
                            self.gamma*b_values[k+1]*(1-int(b_dones[k])) - \
                            b_values[k]
                    a_t += discount*delta
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.Tensor(advantage).to(self.actor.device)

            b_values = T.Tensor(b_values).to(self.actor.device)

            for batch in b_batch:
                states = T.Tensor(b_states[batch])
                old_probs = T.Tensor(b_old_probs[batch])
                actions = T.Tensor(b_actions[batch])

                states = states.to(self.actor.device)
                old_probs = old_probs.to(self.actor.device)
                actions = actions.to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                surr1 = advantage[batch]*prob_ratio
                surr2 = T.clamp(prob_ratio, 1-self.clip, 1+self.clip)
                surr2 *= advantage[batch]

                actor_loss = -T.min(surr1, surr2).mean()

                returns = advantage[batch] + b_values[batch]
                critic_loss = nn.MSELoss()(returns, critic_value)

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear()
