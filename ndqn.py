import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import gym
import numpy as np
from collections import deque


class TransitionMemory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.buffer = deque(maxlen=mem_size)

    def get_all(self, clear=True):
        transitions = map(list, zip(*self.buffer))
        if clear:
            self.buffer.clear()

        return transitions

    def store(self, transition):
        self.buffer.append(transition)


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
    def __init__(self, epsilon, gamma, input_shape, output_shape, n_steps=4):
        self.epsilon = epsilon
        self.gamma = gamma
        self.output_shape = output_shape
        self.n_steps = n_steps

        self.q_eval = DeepQN(input_shape, output_shape, [64, 64])
        self.q_target = DeepQN(input_shape, output_shape, [64, 64])
        self.memory = TransitionMemory(self.n_steps)

        self.tau = 4
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

    def evaluate(self):
        (actions, states, states_, rewards, terminals) = self.memory.get_all(clear=True)

        q_eval = self._evaluate(states, actions)
        q_target = self._evaluate(states_)

        terminals = np.array(terminals)
        rewards = np.array(rewards)

        n_step_rewards = np.zeros_like(rewards)
        gamma_map = [self.gamma ** i for i in range(self.n_steps)]
        for index in range(len(rewards)):
            next_done = np.where(terminals[index:min(len(rewards), index+self.n_steps)] == 1)[0]
            n_step_rollout = rewards[index:min(len(rewards), index+self.n_steps)]

            if len(next_done) != 0:
                n_step_rollout = n_step_rollout[:next_done[0]+1]

            n_step_rewards[index] = (gamma_map[:len(n_step_rollout)] * n_step_rollout).sum()
            n_step_rewards[index] += (self.gamma ** len(n_step_rollout)) * q_target[index]
        rewards = (n_step_rewards - n_step_rewards.mean()) / (n_step_rewards.std() + 1e-5)
        rewards = T.tensor(rewards).float()

        q_target = rewards
        return q_eval, q_target

    def _evaluate(self, states, actions=None):
        if actions:
            indices = np.arange(len(actions))
            return self.q_eval(T.tensor(states).float())[indices, T.tensor(actions).long()]
        else:
            return self.q_target(T.tensor(states).float()).detach().max(axis=1)[0]

    def learn(self, state, action, state_, reward, done):
        self.learn_step += 1
        self.memory.store((action, state, state_, reward, done))

        if self.learn_step % self.n_steps:
            return

        self.q_eval.train()
        q_eval, q_target = self.evaluate()
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
    agent = Agent(1.0, 0.99, env.observation_space.shape, [env.action_space.n],
                  n_steps=15)

    learn(env, agent, 500)
