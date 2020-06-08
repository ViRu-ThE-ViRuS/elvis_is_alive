import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym


class ReplayBuffer:
    def __init__(self, mem_size, input_shape, output_dim):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.mem_counter = 0

        self.rewards = np.zeros(mem_size)
        self.terminals = np.zeros(mem_size)
        self.actions = np.zeros(mem_size)
        self.states = np.zeros((mem_size, *input_shape))
        self.states_ = np.zeros((mem_size, *input_shape))

    def sample(self, batch_size):
        indices = np.random.choice(self.mem_size, batch_size)
        return self.rewards[indices], self.terminals[indices], \
            self.actions[indices], self.states[indices], self.states_[indices]

    def store(self, reward, terminal, action, state, state_):
        index = self.mem_counter % self.mem_size
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.actions[index] = action
        self.states[index] = state
        self.states_[index] = state_

        self.mem_counter += 1


class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, output_dim, layer_dims):
        super(DuelingDDQN, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        layers = [nn.Linear(*input_shape, layer_dims[0])]
        for index, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[index], dim))

        self.A = nn.Linear(layer_dims[-1], output_dim)
        self.V = nn.Linear(layer_dims[-1], 1)

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters())

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        A = self.A(x)
        V = self.V(x)

        return A, V

    def learn(self, values, targets):
        self.optimizer.zero_grad()

        loss = self.loss(input=values, target=targets)
        loss.backward()
        self.optimizer.step()

        return loss


class Agent:
    def __init__(self, epsilon, gamma, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_eval = DuelingDDQN(input_shape, output_dim, [32, 64])
        self.q_next = DuelingDDQN(input_shape, output_dim, [32, 64])
        self.memory = ReplayBuffer(10000, input_shape, output_dim)

        self.batch_size = 64
        self.learn_step = 0

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.output_dim)
        else:
            state = T.tensor([state]).float()
            action, _ = self.q_eval(state)
            return action.max(axis=1)[1].item()

    def _update(self):
        if self.learn_step % 100 == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def sample(self):
        rewards, terminals, actions, states, states_ = \
            self.memory.sample(self.batch_size)

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).view(self.batch_size).float()
        terminals = T.tensor(terminals).view(self.batch_size).long()

        return actions, states, states_, rewards, terminals

    def learn(self, state, action, state_, reward, done):
        if self.memory.mem_counter < self.batch_size:
            self.memory.store(reward, done, action, state, state_)
            return

        self.memory.store(reward, done, action, state, state_)
        self.learn_step += 1
        actions, states, states_, rewards, terminals = self.sample()

        indices = np.arange(self.batch_size)
        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)
        V_s_eval, A_s_eval = self.q_eval(states_)

        q_pred = (V_s + (A_s - A_s.mean(dim=1,
                                        keepdim=True)))[indices, actions]
        q_next = (V_s_ + (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = (V_s_eval + (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        q_target = rewards + self.gamma * \
            q_next[indices, q_eval.max(axis=1)[1]] * (1 - terminals)

        loss = self.q_eval.learn(q_pred, q_target)
        self.epsilon = 0.1 if self.epsilon < 0.1 else self.epsilon * 0.99
        self.counter = 0

        self._update()
        return loss.item()


def learn(env, agent, episodes=500, interval=0.10):
    print('Episode: 5 Episode Mean Reward: Last Loss: 5 Episode Mean Step'
          ' : Last Reward')

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

        if episode % int(episodes * interval) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards[-5:]):5.2f} '
                  f': {losses[-1]: 5.2f}: {np.mean(steps[-5:]): 5.2f} '
                  f': {rewards[-1]: 3f}')

    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(1.0, 1.0,
                  env.observation_space.shape,
                  env.action_space.n)
    learn(env, agent, 500, 0.1)
