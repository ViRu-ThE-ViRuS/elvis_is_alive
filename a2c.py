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

    def get_all(self, clear=True):
        transitions = map(list, zip(*self.buffer))
        if clear:
            self.buffer.clear()

        return transitions

    def store(self, transition):
        self.buffer.append(transition)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActorCriticNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(*input_shape, hidden_layer_dims[0]))
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        for layer in layers:
            T.nn.init.orthogonal_(layer.weight.data)

        self.actor = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.critic = nn.Linear(hidden_layer_dims[-1], 1)

        T.nn.init.orthogonal_(self.actor.weight.data)
        T.nn.init.orthogonal_(self.critic.weight.data)

        self.layers = nn.ModuleList(layers)
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))
        pi = F.softmax(self.actor(states), dim=0)
        v = self.critic(states)

        return pi, v


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(input_shape, output_shape, [64, 64])
        self.memory = TransitionMemory(1000)

    def move(self, state):
        self.actor_critic.eval()
        action_probs, _ = self.actor_critic(T.tensor(state, dtype=T.float))
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def store(self, transition):
        self.memory.store(transition)

    def evaluate(self, clear=True):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=clear)
        log_probs, state_values, dist_entropy = self._evaluate(states, actions)

        discounted_rewards, R = np.zeros_like(rewards), 0
        for index, (reward, done) in enumerate(zip(rewards[::-1], terminals[::-1])):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R * (1 - done)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        rewards = T.tensor(discounted_rewards).float()

        return log_probs, rewards, state_values, dist_entropy

    def _evaluate(self, state, action):
        state = T.tensor(state).float()
        action = T.tensor(action).float()

        action_probs, state_value = self.actor_critic(state)
        dist = T.distributions.Categorical(action_probs)

        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return log_prob, T.squeeze(state_value), dist_entropy

    def learn(self):
        self.actor_critic.train()

        log_probs, rewards, state_values, dist_entropy = self.evaluate(clear=True)
        advantage = rewards - state_values

        actor_loss = -log_probs * advantage.detach()
        critic_loss = advantage ** 2
        entropy_loss = -dist_entropy * 0.001
        loss = (actor_loss + critic_loss + entropy_loss).mean()

        self.actor_critic.optimizer.zero_grad()
        loss.backward()
        self.actor_critic.optimizer.step()

        # visualize
        # make_dot(loss, params=dict(self.actor_critic.named_parameters())).render("attached")

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
            agent.store((action, state, state_, reward, done))

            state = state_
            total_reward += reward
            n_steps += 1

        loss = agent.learn()
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
    agent = Agent(0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 500)
