import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import gym
import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size, input_shape, output_shape):
        self.mem_counter = 0
        self.mem_size = mem_size
        self.input_shape = input_shape

        self.actions = np.zeros(mem_size)
        self.states = np.zeros((mem_size, *input_shape))
        self.states_ = np.zeros((mem_size, *input_shape))
        self.rewards = np.zeros(mem_size)
        self.terminals = np.zeros(mem_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.mem_size, batch_size)
        return self.actions[indices], self.states[indices], \
            self.states_[indices], self.rewards[indices], \
            self.terminals[indices]

    def store(self, action, state, state_, reward, terminal):
        index = self.mem_counter % self.mem_size

        self.actions[index] = action
        self.states[index] = state
        self.states_[index] = state_
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.mem_counter += 1


class DeepQN(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(DeepQN, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

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

        self.q_eval = DeepQN(input_shape, output_shape, [64, 64])
        self.q_target = DeepQN(input_shape, output_shape, [64, 64])
        self.memory = ReplayBuffer(10000, input_shape, output_shape)

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

    def sample(self):
        actions, states, states_, rewards, terminals = \
            self.memory.sample(self.batch_size)

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).view(self.batch_size).float()
        terminals = T.tensor(terminals).view(self.batch_size).long()

        return actions, states, states_, rewards, terminals

    def learn(self, state, action, state_, reward, done):
        self.memory.store(action, state, state_, reward, done)

        if self.memory.mem_counter < self.batch_size:
            return

        self.q_eval.train()
        self.learn_step += 1
        actions, states, states_, rewards, terminals = self.sample()
        indices = np.arange(self.batch_size)
        q_eval = self.q_eval(states)[indices, actions]
        target_actions = self.q_eval(states_).detach().max(axis=1)[1]
        q_target = self.q_target(states_).detach()[indices, target_actions]
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
            print(f'{episode:5d} : {np.mean(rewards):5.2f} '
                  f': {np.mean(losses):5.3f}: {np.mean(steps):5.2f}')
            rewards = []
            losses = [0]
            steps = []

    print(f'{episode:5d} : {np.mean(rewards):5.2f} '
          f': {np.mean(losses):5.3f}: {np.mean(steps):5.2f}')
    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    agent = Agent(1.0, 0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 1000)
