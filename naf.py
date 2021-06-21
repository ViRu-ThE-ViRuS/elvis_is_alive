import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return T.tensor(self.state * self.scale).float()


class ReplayBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.buffer = deque(maxlen=mem_size)

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        sample_indices = np.random.choice(len(self.buffer), sample_size)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        return map(list, zip(*samples))

    def store(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)


class NAF_Net(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(NAF_Net, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        self.layers = nn.ModuleList(layers)
        self.mu = nn.Linear(hidden_layer_dims[-1], output_shape)
        self.v = nn.Linear(hidden_layer_dims[-1], 1)
        self.L = nn.Linear(hidden_layer_dims[-1], output_shape ** 2)
        self._initialize_layers()

        self.tril_mask = T.tril(T.ones(output_shape, output_shape), diagonal=-1).unsqueeze(0)
        self.diag_mask = T.diag(T.diag(T.ones(output_shape, output_shape))).unsqueeze(0)

    def _initialize_layers(self):
        for layer in self.layers:
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)

        self.mu.weight.data.mul_(0.1)
        self.v.weight.data.mul_(0.1)
        self.L.weight.data.mul_(0.1)

        self.mu.bias.data.mul_(0.1)
        self.v.bias.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

    def forward(self, states, actions=None):
        x = states
        for layer in self.layers:
            x = T.tanh(layer(x))

        mu = T.tanh(self.mu(x))
        V = self.v(x)

        Q = None
        if actions is not None:
            L = self.L(x).view(-1, self.output_shape, self.output_shape)
            L = L * self.tril_mask.expand_as(L) + T.exp(L) * self.diag_mask.expand_as(L)
            P = T.bmm(L, L.transpose(2, 1))

            u_mu = (actions - mu).unsqueeze(-1)
            A = -0.5 * T.bmm(T.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]
            Q = A + V

        return mu, Q, V


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape):
        self.gamma = gamma
        self.lr = 0.001
        self.tau = 0.05
        self.network_params = [64, 64]
        self.batch_size = 64
        self.explore_limit = 200
        self.max_grad_norm = 0.5

        self.noise = OUNoise(output_shape)
        self.policy = NAF_Net(input_shape, output_shape, self.network_params)
        self.policy_old = NAF_Net(input_shape, output_shape, self.network_params)
        self.optimizer = T.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(10000)

        self.learn_step = 0
        self._initialize()

    def move(self, state):
        self.policy.eval()

        with T.no_grad():
            mu, _, _ = self.policy(T.tensor(state).float())

            if self.learn_step < self.explore_limit:
                mu += T.Tensor(self.noise())

        return mu.clamp(-1, 1).numpy()

    def store(self, transition):
        self.memory.store(transition)

    def _initialize(self):
        for target_param, param in zip(self.policy_old.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(param.data)

    def update(self):
        for target_param, param in zip(self.policy_old.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def evaluate(self):
        (states, actions, states_, rewards, terminals) = self.memory.sample(self.batch_size)

        states = T.tensor(states).float()
        actions = T.tensor(actions).long()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float().view(-1, 1)
        terminals = T.tensor(terminals).long().view(-1, 1)

        return states, actions, states_, rewards, terminals

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, states_, rewards, terminals = self.evaluate()

        self.policy.train()
        self.learn_step += 1

        _, advantages, _ = self.policy(states, actions)
        _, _, state_values_ = self.policy_old(states_, None)
        targets = rewards + self.gamma * state_values_.detach() * (1 - terminals)
        loss = F.mse_loss(advantages, targets)

        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # visualize
        # make_dot(loss, params=dict(self.policy.named_parameters())).render("attached")
        # raise SystemError

        self.update()
        return loss.item()


def learn(env, agent, episodes=500):
    print('Episode: Mean Reward: Mean Loss: Mean Step')

    rewards = []
    losses = [0]
    steps = []
    num_episodes = episodes
    timestep = 0
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            timestep += 1
            action = agent.move(state)
            state_, reward, done, _ = env.step(action)
            agent.store((state, action, state_, reward, done))

            state = state_
            total_reward += reward
            n_steps += 1

            loss = agent.learn()
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
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(0.99, env.observation_space.shape, env.action_space.shape[0])
    learn(env, agent, 100)
