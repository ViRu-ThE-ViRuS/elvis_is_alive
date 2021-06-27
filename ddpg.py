import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)

    def __call__(self):
        self.x_prev = x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        return T.tensor(x)


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


# TODO(vir): add batchnorm layers, for paper coherence
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(CriticNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))
        layers.append(nn.Linear(*output_shape, 1))

        for layer in layers:
            _l = 1/np.sqrt(layer.weight.data.size()[0])
            T.nn.init.uniform_(layer.weight.data, -_l, _l)
            T.nn.init.uniform_(layer.bias.data, -_l, _l)

        self.layers = nn.ModuleList(layers)

    def forward(self, states, actions):
        for layer in self.layers[:-2]:
            states = F.relu(layer(states))
        states = F.relu(self.layers[-2](states) + actions)
        state_values = self.layers[-1](states)

        return state_values


# TODO(vir): add batchnorm layers, for paper coherence
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActorNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        for layer in layers:
            _l = 1/np.sqrt(layer.weight.data.size()[0])
            T.nn.init.uniform_(layer.weight.data, -_l, _l)
            T.nn.init.uniform_(layer.bias.data, -_l, _l)

        self.layers = nn.ModuleList(layers)

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return T.tanh(self.layers[-1](states))


class Agent:
    def __init__(self, env_name, gamma, lrs=(0.001, 0.001)):
        self.env = gym.make(env_name)

        self.lrs = lrs
        self.gamma = gamma

        self.tau = 0.001
        self.batch_size = 32
        self.max_grad_norm = 0.5
        self.actor_network_params = [128, 128]
        self.critic_network_params = [128, 128]

        self.memory = ReplayBuffer(10000)
        self.noise = OUActionNoise(np.zeros(self.env.action_space.shape))

        self.actor = ActorNetwork(self.env.observation_space.shape,
                                  self.env.action_space.shape,
                                  self.actor_network_params)
        self.target_actor = ActorNetwork(self.env.observation_space.shape,
                                         self.env.action_space.shape,
                                         self.actor_network_params)

        self.critic = CriticNetwork(self.env.observation_space.shape,
                                    self.env.action_space.shape,
                                    self.critic_network_params)
        self.target_critic = CriticNetwork(self.env.observation_space.shape,
                                           self.env.action_space.shape,
                                           self.critic_network_params)

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), self.lrs[0])
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), self.lrs[1])

        self.update()

    def update(self):
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.eval()

    def store(self, transition):
        self.memory.store(transition)

    def move(self, state):
        mu = self.actor(T.tensor(state).float())
        mu_prime = mu + self.noise()
        return mu_prime.detach().numpy()

    def sample(self):
        (states, actions, states_, rewards, dones) = self.memory.sample(self.batch_size)

        states = T.tensor(states).float()
        actions = T.tensor(actions).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float()
        dones = T.tensor(dones).long()

        return states, actions, states_, rewards, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.actor.train()
        states, actions, states_, rewards, terminals = self.sample()

        with T.no_grad():
            actions_ = self.target_actor(states_)
            state_values_ = self.target_critic(states_, actions_).flatten()

        state_values = self.critic(states, actions).flatten()
        target_values = rewards + self.gamma * state_values_ * (1 - terminals)
        target_values = (target_values - target_values.mean()) / (target_values.std() + 1e-5)

        critic_loss = F.mse_loss(input=state_values, target=target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # actor maximises the critic function to selection optimal action
        # actor(x) = argmax_a critic(x, a)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # visualize
        # make_dot(critic_loss, params=dict(self.critic.named_parameters())).render("attached")
        # raise SystemError

        self.update()
        return (actor_loss + critic_loss).item()


def learn(agent, episodes=500):
    print('Episode: Mean Reward: Mean Loss: Mean Step')

    env = agent.env
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
    agent = Agent('LunarLanderContinuous-v2', 0.99, (0.0001, 0.0001))
    learn(agent, 100)
