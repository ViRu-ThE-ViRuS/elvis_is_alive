import torch as T
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np


class DeepSarsaAgent(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(DeepSarsaAgent, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = []
        layers.append(nn.Linear(*input_shape, hidden_layer_dims[0]))
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        self.layers = nn.ModuleList(layers)

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters())

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return self.layers[-1](states)

    def learn(self, predictions, targets):
        self.optimizer.zero_grad()
        loss = self.loss(input=predictions, target=targets)
        loss.backward()
        self.optimizer.step()

        return loss


class Agent:
    def __init__(self, epsilon, gamma, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_eval = DeepSarsaAgent(input_shape, output_shape, [64, 128])

        self.learn_step = 0

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(*self.output_shape)
        else:
            state = T.tensor([state]).float()
            action = self.q_eval(state).max(axis=1)[1]
            return action.item()

    def learn(self, state, action, state_, reward, done):
        self.learn_step += 1
        action = T.tensor(action).long()
        state = T.tensor(state).float()
        state_ = T.tensor(state_).float()
        reward = T.tensor(reward).float()
        terminal = T.tensor(done).long()

        q_eval = self.q_eval(state)[action]
        q_next = self.q_eval(state_)
        action_next = q_next.max(axis=0)[1]
        q_target = reward + self.gamma * q_next[action_next] * (1 - terminal)
        loss = self.q_eval.learn(q_eval, q_target)

        return loss.item()


def learn(env, agent, episodes=500):
    print('Episode: 5 Episode Mean Reward: Last Loss: 5 Episode Mean Step'
          ' : Last Reward')

    rewards = []
    losses = [0]
    steps = []
    num_episodes = episodes
    for episode in range(num_episodes+1):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            action = agent.move(state)
            state_, reward, done, _ = env.step(action)
            loss = agent.learn(state, action, state_, reward, done)

            state = state_
            total_reward += reward
            n_steps += 1

            if loss:
                losses.append(loss)

        rewards.append(total_reward)
        steps.append(n_steps)

        agent.epsilon *= 0.9 if agent.epsilon > 0.10 else 1.0
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

    learn(env, agent, 1000)
