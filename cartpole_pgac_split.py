import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym


class GenericNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(GenericNetwork, self).__init__()

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


class SplitAgent(object):
    def __init__(self, gamma, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma

        self.actor = GenericNetwork(input_shape, output_shape, [64, 128])
        self.critic = GenericNetwork(input_shape, [1], [64, 128])
        self.log_probs = None

    def move(self, state):
        action_probs = F.softmax(self.actor(T.tensor(state, dtype=T.float)),
                                 dim=0)
        distribution = T.distributions.Categorical(action_probs)
        action_taken = distribution.sample()
        self.log_probs = distribution.log_prob(action_taken)

        return action_taken.item()

    def learn(self, state, state_, reward, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic(T.tensor(state_, dtype=T.float))
        critic_value = self.critic(T.tensor(state, dtype=T.float))
        reward = T.tensor(reward, dtype=T.float)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) \
            - critic_value
        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        loss = actor_loss + critic_loss
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

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
            loss = agent.learn(state, state_, reward, done)
            state = state_
            total_reward += reward
            n_steps += 1

            losses.append(loss)

        rewards.append(total_reward)
        steps.append(n_steps)

        if episode % (episodes//10) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards[-5:]):5.2f} '
                  f': {losses[-1]: 5.2f}: {np.mean(steps[-5:]): 5.2f} '
                  f': {rewards[-1]: 3f}')

    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = SplitAgent(1.0,
                       env.observation_space.shape,
                       [env.action_space.n])

    learn(env, agent, 500)
