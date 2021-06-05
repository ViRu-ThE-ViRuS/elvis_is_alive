import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import gym
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.buffer = deque(maxlen=mem_size)

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        sample_indices = np.random.choice(len(self.buffer), sample_size)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        return map(list, zip(*samples))

    def get_all(self, reverse=True):
        return map(list, zip(*reversed(self.buffer))) if reverse else map(list, zip(*self.buffer))

    def store(self, transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer = deque(maxlen=self.mem_size)


class DeepQN(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(DeepQN, self).__init__()

        layers = []
        layers.append(nn.Linear(*input_shape, hidden_layer_dims[0]))
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return self.layers[-1](states)

    def learn(self, predictions, targets):
        loss = self.loss(input=predictions, target=targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class Agent:
    def __init__(self, epsilon, gamma, input_shape, output_shape):
        self.epsilon = epsilon
        self.gamma = gamma
        self.output_shape = output_shape
        self.n_step = 16

        self.q_eval = DeepQN(input_shape, output_shape, [64, 64])
        self.q_target = DeepQN(input_shape, output_shape, [64, 64])
        self.memory = ReplayBuffer(self.n_step)

        self.tau = 8
        self.batch_size = 32
        self.learn_step = 0

        self.update()

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(*self.output_shape)
        else:
            self.q_eval.eval()
            state = T.tensor([state]).float()
            action = self.q_eval(state).max(axis=1)[1]
            return action.item()

    def update(self):
        if self.learn_step % self.tau == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            self.q_target.eval()

    def get_all(self):
        (actions, states, states_, rewards, terminals) = self.memory.get_all(reverse=True)
        self.memory.clear()

        discounted_rewards = np.zeros(len(rewards),)
        _reward = 0
        for index, (reward, terminal) in enumerate(zip(rewards, terminals)):
            _reward = discounted_rewards[len(rewards) - index - 1] = reward * (1 - terminal) + self.gamma * _reward

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(discounted_rewards).float()
        terminals = T.tensor(terminals).long()

        return actions, states, states_, rewards, terminals

    def learn(self, state, action, state_, reward, done):
        self.learn_step += 1
        self.memory.store((action, state, state_, reward, done))

        if self.learn_step % self.n_step:
            return

        self.q_eval.train()
        actions, states, states_, rewards, terminals = self.get_all()
        indices = np.arange(len(actions))
        q_eval = self.q_eval(states)[indices, actions]
        q_target = self.q_target(states_).detach().max(axis=1)[0]
        q_target = rewards + self.gamma * q_target * (1 - terminals)

        loss = self.q_eval.learn(q_eval, q_target)
        self.epsilon *= 0.95 if self.epsilon > 0.1 else 1.0

        # visualize
        # make_dot(loss, params=dict(self.q_eval.named_parameters())).render("attached")

        self.update()
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
            loss = agent.learn(state, action, state_, reward, done)

            state = state_
            total_reward += reward
            n_steps += 1

            if loss:
                losses.append(loss)

        rewards.append(total_reward)
        steps.append(n_steps)

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
    agent = Agent(1.0, 0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 5000)
