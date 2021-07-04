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

    def store(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)


class CategoricalDeepQN(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_layer_dims,
                 n_atoms=51, V_MAX=-10, V_MIN=+10):
        super(CategoricalDeepQN, self).__init__()

        self.n_atoms = n_atoms
        self.n_actions = n_outputs

        self.V_MIN = V_MIN
        self.V_MAX = V_MAX
        self.DZ = (V_MAX - V_MIN) / (n_atoms - 1)

        layers = [nn.Linear(n_inputs, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], self.n_actions * self.n_atoms))

        self.layers = nn.ModuleList(layers)
        self.register_buffer('supports', T.arange(V_MIN, V_MAX + self.DZ, self.DZ))

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        z = self.layers[-1](states)
        p = F.softmax(z.view(-1, self.n_actions, self.n_atoms), dim=2)
        return p

    def move(self, states):
        with T.no_grad():
            prob = self.forward(states)
            expected_value = prob * self.supports
            actions = T.sum(expected_value, dim=2)
        return actions, prob


class Agent:
    def __init__(self, epsilon, gamma, observation_space, action_space, reward_space):
        self.epsilon = epsilon
        self.gamma = gamma

        self.reward_space = reward_space
        self.action_space = action_space
        self.n_inputs = observation_space.shape[0]
        self.n_actions = action_space.n

        self.lr = 0.0001
        self.n_atoms = 51
        self.V_MIN, self.V_MAX = -500, 500  # TODO(vir): tune this from `reward_space`

        self.q_eval = CategoricalDeepQN(self.n_inputs, self.n_actions, [64, 64],
                                        self.n_atoms, self.V_MAX, self.V_MIN)
        self.q_target = CategoricalDeepQN(self.n_inputs, self.n_actions, [64, 64],
                                          self.n_atoms, self.V_MAX, self.V_MIN)

        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(10000)

        self.learn_step = 0
        self.tau = 0.01
        self.batch_size = 128

        self.update(soft=False)

    def move(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            actions, _ = self.q_eval.move(T.tensor(state).float())
            action = T.argmax(actions, dim=1)
            return action.item()

    def update(self, soft=True):
        if soft:
            tau = self.tau
        else:
            tau = 1.0

        for target_param, param in zip(self.q_target.parameters(), self.q_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        self.q_target.eval()
        self.q_eval.eval()

    def sample(self):
        (actions, states, states_, rewards, terminals) = \
            self.memory.sample(self.batch_size)

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float().view(-1, 1)
        terminals = T.tensor(terminals).long().view(-1, 1)

        return actions, states, states_, rewards, terminals

    def projected_dist(self, next_dist, states_, rewards, terminals):
        delta_z = float(self.V_MAX - self.V_MIN) / (self.n_atoms - 1)
        support = T.linspace(self.V_MIN, self.V_MAX, self.n_atoms).unsqueeze(0).expand_as(next_dist)

        rewards = rewards.expand_as(next_dist)
        terminals = terminals.expand_as(next_dist)

        Tz = (rewards + self.gamma * support * (1 - terminals)).clamp(min=self.V_MIN, max=self.V_MAX)
        b = ((Tz - self.V_MIN) / delta_z)

        l = b.floor().long()
        u = b.ceil().long()
        offset = T.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long()\
            .unsqueeze(1).expand(self.batch_size, self.n_atoms)

        proj_dist = T.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def learn(self, state, action, state_, reward, done):
        self.memory.store((action, state, state_, reward, done))

        if len(self.memory) < self.batch_size:
            return

        self.learn_step += 1
        self.q_eval.train()

        indices = np.arange(self.batch_size)
        actions, states, states_, rewards, terminals = self.sample()

        with T.no_grad():
            actions_, dist_next = self.q_target.move(states_)
            actions_dist_ = dist_next[indices, actions_.max(dim=1)[1]]
            projected_dist = self.projected_dist(actions_dist_, states_, rewards, terminals)

        prob_dist = self.q_eval(states)[indices, actions]
        loss = -(projected_dist * prob_dist.log()).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # visualize
        # make_dot(loss, params=dict(self.q_eval.named_parameters())).render("attached")
        # raise SystemError

        self.epsilon *= 0.95 if self.epsilon > 0.1 else 1.0
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
    agent = Agent(1.0, 0.9, env.observation_space, env.action_space, env.reward_range)
    learn(env, agent, 500)
