import gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class PolicyGradient(object):
    def __init__(self, n_actions: int, n_features: int, 
            lr: float =0.01, reward_decay: float =0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay

        self.ep_o = []
        self.ep_a = []
        self.ep_r = []

        self.net = Net(n_features, n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # nn.CrossEntropy = nn.LogSoftmax + nn.NLLLoss
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def choose_action(self, o):
        o_tensor = torch.tensor(o, dtype=torch.float).unsqueeze(0)
        prob = nn.Softmax(dim=1)(self.net(o_tensor))
        m = Categorical(prob)
        action = np.arange(prob.shape[1])[m.sample().numpy()[0]]

        return action

    def store_transition(self, o, a, r):
        self.ep_o.append(o)
        self.ep_a.append(a)
        self.ep_r.append(r)


    def learn(self):
        o_tensor = torch.tensor(np.vstack(self.ep_o), dtype=torch.float)
        a_tensor = torch.tensor(np.array(self.ep_a), dtype=torch.long)
        vt = self.discount_and_norm_rewards()
        vt_tensor = torch.tensor(np.array(vt), dtype=torch.float)

        # Can’t call numpy() on Variable that requires grad. Use var.detach().numpy()
        # means some of data are not as torch.tensor type

        pred = self.net(o_tensor)

        neg_log_prob = self.criterion(pred, a_tensor)
        loss = torch.mean(neg_log_prob*vt_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_o, self.ep_a, self.ep_r = [], [] ,[]

        return vt

    def discount_and_norm_rewards(self):
        discounted_ep_r = np.zeros_like(self.ep_r)
        running_add = 0
        for t in reversed(range(len(self.ep_r))):
            running_add = running_add*self.gamma + self.ep_r[t]
            discounted_ep_r[t] = running_add

        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)

        return discounted_ep_r

def main():
    RENDER = False

    # DISPLAY_REWARD_THRESHOLD = 195
    # env = gym.make('CartPole-v0')
    
    DISPLAY_REWARD_THRESHOLD = -110
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)


    agent = PolicyGradient(
            n_actions = env.action_space.n,
            n_features = env.observation_space.shape[0],
            lr=0.02,
            reward_decay=0.99
            )

    episodes = 3000
    total_reward = [0]*episodes
    real_reward = [0]*episodes
    for episode in range(episodes):
        s = env.reset()
        while True:
            # if RENDER:
                # env.render()
            env.render()

            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            real_reward[episode] += r

            position, velocity = s_
            print(s_)

            position_r = abs(position+0.5)
            if (velocity<0.0 and a==0) or (velocity>=0.0 and a==2):
                action_r = 1
            else:
                action_r = 0

            r = position_r + action_r
            agent.store_transition(s, a, r)

            if done:
                ep_rs_sum = sum(agent.ep_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward*0.99+ep_rs_sum*0.01

                total_reward[episode] = running_reward
                if sum(real_reward)/(episode+1) > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print(f"Episode: {episode}, reward: {running_reward}, avg reward = {sum(real_reward)/(episode+1):.2f}")

                vt = agent.learn()

                if episode == 30:
                    plt.plot(vt)
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break
            s = s_
                    

if __name__ == "__main__":
    main()
