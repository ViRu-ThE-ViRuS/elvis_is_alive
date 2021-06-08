import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import gym
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(PolicyNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(*input_shape, hidden_layer_dims[0]))
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        self.layers = nn.ModuleList(layers)
        self.optimizer = T.optim.Adam(self.parameters())

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return F.softmax(self.layers[-1](states), dim=0)


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape):
        self.gamma = gamma

        self.policy = PolicyNetwork(input_shape, output_shape, [64, 64])
        self.action_memory = []
        self.reward_memory = []

    def move(self, state):
        self.policy.eval()
        action_probs = self.policy(T.tensor(state, dtype=T.float))
        distribution = T.distributions.Categorical(action_probs)
        action_taken = distribution.sample()
        log_prob = distribution.log_prob(action_taken)

        self.action_memory.append(log_prob)
        return action_taken.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.train()
        self.policy.optimizer.zero_grad()

        G, G_current = np.zeros_like(self.reward_memory), 0
        for index, step in reversed(list(enumerate(self.reward_memory))):
            G_current += step * (self.gamma ** (index-1))
            G[index] = G_current

        G = (G - np.mean(G)) / (np.std(G) if np.std(G) != 0 else 1)
        G = T.tensor(G)

        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob
        loss.backward()

        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []

        return loss.item()


def learn(env, agent, episodes=500):
    print('Episode: Mean Reward: Mean Loss: Mean Step')

    rewards = []
    losses = [0]
    steps = []
    num_episodes = episodes
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            action = agent.move(state)
            state_, reward, done, _ = env.step(action)
            agent.store_reward(reward)

            state = state_
            total_reward += reward
            n_steps += 1

        loss = agent.learn()
        rewards.append(total_reward)
        steps.append(n_steps)
        losses.append(loss)

        if episode % (episodes // 10) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards):06.2f} '
                  f': {np.mean(losses):06.4f} : {np.mean(steps):06.2f}')
            rewards = []
            losses = [0]
            steps = []

    print(f'{episode:5d} : {np.mean(rewards):06.2f} '
          f': {np.mean(losses):06.4f} : {np.mean(steps):06.2f}')
    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    agent = Agent(0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 500)
