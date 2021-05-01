# import os
import torch as T
from torch.nn import functional as F
# import numpy as np
from buffer import ReplayBuffer
from networks import (
    ActorNetwork,
    CriticNetwork,
    ValueNetwork
)


class Agent:

    def __init__(self, alpha=3e-4, beta=3e-4, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000,
                 tau=5e-3, fc1_dim=256, fc2_dim=256, batch_size=256,
                 reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions,
                                  env.action_space.high)
        self.critic1 = CriticNetwork(beta, input_dims, n_actions,
                                     name='critic1')
        self.critic2 = CriticNetwork(beta, input_dims, n_actions,
                                     name='critic2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        state = T.Tensor([obs]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, n_state, action, reward, done):
        self.memory.store_transition(state, n_state, action, reward, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        trg_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        trg_value_state_dict = dict(trg_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                                     (1-tau)*trg_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_ckpt()
        self.value.save_ckpt()
        self.target_value.save_ckpt()
        self.critic1.save_ckpt()
        self.critic2.save_ckpt()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_ckpt()
        self.value.load_ckpt()
        self.target_value.load_ckpt()
        self.critic1.load_ckpt()
        self.critic2.load_ckpt()

    def learn(self):
        if self.memory.mem_ptr < self.batch_size:
            return

        s, ns, a, r, t = \
            self.memory.sample_buffer(self.batch_size)

        s = T.Tensor(s).to(self.actor.device)
        ns = T.Tensor(ns).to(self.actor.device)
        a = T.Tensor(a).to(self.actor.device)
        r = T.Tensor(r).to(self.actor.device)
        t = T.tensor(t).to(self.actor.device)

        # update value net
        value = self.value(s).view(-1)
        value_ = self.target_value(ns).view(-1)
        value_[t] = 0.0

        actions, logprobs = self.actor.sample_normal(s, reparameterize=False)
        logprobs = logprobs.view(-1)
        critic_value = T.min(self.critic1(s, actions),
                             self.critic2(s, actions))
        critic_value = critic_value.view(-1)

        value_target = critic_value - logprobs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # update actor net
        actions, logprobs = self.actor.sample_normal(s, reparameterize=True)
        logprobs = logprobs.view(-1)
        critic_value = T.min(self.critic1(s, actions),
                             self.critic2(s, actions))
        critic_value = critic_value.view(-1)

        actor_loss = T.mean(logprobs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # update critic net
        q_hat = self.scale * r + self.gamma * value_
        q1 = self.critic1(s, a).view(-1)
        q2 = self.critic2(s, a).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q_hat, q1)
        critic2_loss = 0.5 * F.mse_loss(q_hat, q2)
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()
