import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


class ReplayBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.buffer = deque(maxlen=mem_size)

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        sample_indices = np.random.choice(len(self.buffer), sample_size)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        return map(np.array, zip(*samples))

    def store(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims, alpha=4):
        super(SoftQNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        self.layers = nn.ModuleList(layers)
        self.q = nn.Linear(hidden_layer_dims[-1], *output_shape)

        self.alpha = alpha

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return self.q(states)

    def value(self, q):
        return self.alpha * T.log(T.sum(T.exp(q / self.alpha), dim=1, keepdim=True))

    def sample_action(self, states):
        with T.no_grad():
            q = self.forward(states)
            v = self.value(q).squeeze()

            # softmax
            pi = T.exp((q-v) / self.alpha)
            pi = pi / T.sum(pi)

            dist = T.distributions.Categorical(pi)
            action = dist.sample()

        return action.item()


# TODO(vir): add continuous action space version
class AgentDiscrete:
    def __init__(self, epsilon, gamma, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = 0.003
        self.batch_size = 16
        self.K = 4
        self.tau = 10
        self.learn_step = 0

        self.q_eval = SoftQNetwork(input_shape, output_shape, [64, 64])
        self.q_target = SoftQNetwork(input_shape, output_shape, [64, 64])
        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(10000)

        self.update()

    def update(self):
        if self.learn_step % self.tau == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            self.q_target.eval()
            self.q_eval.eval()

    def move(self, state):
        return self.q_eval.sample_action(T.tensor([state]).float())

    def sample(self):
        (actions, states, states_, rewards, terminals) = self.memory.sample(self.batch_size)

        actions = T.tensor(actions).long().view(-1, 1)
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float().view(-1, 1)
        terminals = T.tensor(terminals).long().view(-1, 1)

        return actions, states, states_, rewards, terminals

    def learn(self, state, action, state_, reward, done):
        self.memory.store((action, state, state_, reward, done))
        if len(self.memory) < self.batch_size:
            return

        self.learn_step += 1
        self.q_eval.train()

        actions, states, states_, rewards, terminals = self.sample()

        with T.no_grad():
            q_target_ = self.q_target(states_)
            state_values_ = self.q_target.value(q_target_)
            q_next = rewards + self.gamma * state_values_ * (1 - terminals)

        state_values = self.q_eval(states).gather(1, actions)
        loss = F.mse_loss(state_values, q_next)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    agent = AgentDiscrete(1.0, 0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 500)
