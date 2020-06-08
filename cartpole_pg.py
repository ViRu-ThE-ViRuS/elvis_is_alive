import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(PolicyNetwork, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

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
        return self.layers[-1](states)


class Agent(object):
    def __init__(self, epsilon, gamma, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epsilon = epsilon
        self.gamma = gamma

        self.policy = PolicyNetwork(input_shape, output_shape, [64, 128])
        self.action_memory = []
        self.reward_memory = []

    def move(self, state):
        action_probs = F.softmax(self.policy(T.tensor(state, dtype=T.float)),
                                 dim=0)
        distribution = T.distributions.Categorical(action_probs)
        action_taken = distribution.sample()
        log_probs = distribution.log_prob(action_taken)

        self.action_memory.append(log_probs)
        return action_taken.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory)

        G_current = 0
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
    print('Episode: 5 Episode Mean Reward: Last Loss: 5 Episode Mean Step'
          ' : Last Reward')

    rewards = []
    losses = []
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

        if episode % (episodes//10) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards[-5:]):5.2f} '
                  f': {losses[-1]: 5.2f}: {np.mean(steps[-5:]): 5.2f} '
                  f': {rewards[-1]: 3f}')

    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(1.0, 1.0,
                  env.observation_space.shape,
                  [env.action_space.n])

    learn(env, agent, 500)
