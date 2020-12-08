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

        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())

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
        # state_value.shape = (batch_size, 1)

        # action_logprob.shape = (batch_size,)
        # state_value.shape = (batch_size, 1)
        # dist_entropy.shape = (batch_size,)
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
            action = self.policy_old.act(state, memory).cpu().numpy()[0]
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
        # rewards.shape = (mem_len,)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # rewards.shape = (mem_len,)

        old_states = torch.stack(memory.states).to(self.device).squeeze(1)
        # old_states.shape = (mem_len, s_dim)
        old_actions = torch.stack(memory.actions).to(self.device).squeeze(1)
        # old_actions.shape = (mem_len, a_dim)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).squeeze(1)
        # old_logprobs.shape = (mem_len,)

        # total_loss = 0

        for i in range(self.K_epochs):
            result = self.policy.evaluate(old_states, old_actions)
            logprobs, state_values, dist_entropy = result
            # logprob.shape = (mem_len,)
            # state_value.shape = (mem_len,)
            # dist_entropy.shape = (mem_len,)

            advantages = rewards - state_values.detach()
            # advantages.shape = (mem_len,)

            ratios = torch.exp(logprobs - old_logprobs)
            # ratios.shape = (mem_len,)

            surr1 = advantages * ratios
            # surr1.shape = (mem_len,)
            surr2 = advantages * torch.clip(ratios,
                                            1-self.eps_clip,
                                            1+self.eps_clip)
            # surr2.shape = (mem_len,)

            loss = (-torch.min(surr1, surr2) +
                    0.5 * self.criterion(state_values, rewards) -
                    0.01 * dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # total_loss += loss.mean().item()

        # mean_loss = total_loss/self.K_epochs

        self.policy_old.load_state_dict(self.policy.state_dict())
