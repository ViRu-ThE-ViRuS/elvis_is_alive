import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


class TransitionMemory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.buffer = deque(maxlen=mem_size)

    def get_all(self, reverse=True):
        return map(list, zip(*reversed(self.buffer))) if reverse else map(list, zip(*self.buffer))

    def store(self, transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer = deque(maxlen=self.mem_size)


class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, output_shape, layer_dims):
        super(DuelingDDQN, self).__init__()

        layers = []
        layers.append(nn.Linear(*input_shape, layer_dims[0]))
        for index, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[index], dim))

        self.A = nn.Linear(layer_dims[-1], *output_shape)
        self.V = nn.Linear(layer_dims[-1], 1)

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        A = self.A(x)
        V = self.V(x)
        return A, V

    def get_q_table(self, states):
        A, V = self.forward(states)
        return V + (A - A.mean(dim=1, keepdim=True))

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

        self.q_eval = DuelingDDQN(input_shape, output_shape, [64, 64])
        self.q_next = DuelingDDQN(input_shape, output_shape, [64, 64])
        self.memory = TransitionMemory(self.n_step)

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
            action, _ = self.q_eval(state)
            return action.max(axis=1)[1].item()

    def update(self):
        if self.learn_step % self.tau == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            self.q_next.eval()

    def get_all(self):
        (actions, states, states_, rewards, terminals) = self.memory.get_all(reverse=True)
        self.memory.clear()

        discounted_rewards, _reward = np.zeros(len(rewards)), 0
        for index, reward in enumerate(rewards):
            _reward = discounted_rewards[index] = reward + self.gamma * _reward

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
        q_pred = self.q_eval.get_q_table(states)[indices, actions]
        target_actions = self.q_eval.get_q_table(states_).detach().max(axis=1)[1]
        q_target = self.q_next.get_q_table(states_).detach()[indices, target_actions]
        q_target = rewards + self.gamma * q_target * (1 - terminals)

        loss = self.q_eval.learn(q_pred, q_target)
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

    learn(env, agent, 2000)
