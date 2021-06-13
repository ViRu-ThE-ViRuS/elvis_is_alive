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


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(PolicyNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        self.layers = nn.ModuleList(layers)
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return F.softmax(self.layers[-1](states), dim=0)

    def evaluate(self, states, actions):
        action_probs = self.forward(states)
        dist = T.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape, update_interval=2000, K=10):
        self.gamma = gamma
        self.update_interval = update_interval
        self.K = K

        self.policy = PolicyNetwork(input_shape, output_shape, [64, 64])
        self.policy_old = PolicyNetwork(input_shape, output_shape, [64, 64])
        self.memory = TransitionMemory(update_interval)

        self.update()

    def move(self, state):
        action_probs = self.policy_old(T.tensor(state, dtype=T.float))
        action_taken = T.distributions.Categorical(action_probs).sample()
        return action_taken.item()

    def evaluate(self, clear=True):
        (states, actions, rewards, dones) = self.memory.get_all(clear)

        discounted_rewards, R = np.zeros_like(rewards), 0
        for index, (reward, done) in enumerate(zip(rewards[::-1], dones[::-1])):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R * (1 - done)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        states = T.tensor(states).float()
        actions = T.tensor(actions).float()
        discounted_rewards = T.tensor(discounted_rewards).float()

        log_probs, _ = self.policy_old.evaluate(states, actions)
        return states, actions, discounted_rewards, dones, log_probs

    def store(self, transition):
        self.memory.store(transition)

    def update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

    def learn(self):
        self.policy.train()

        losses = []
        states, actions, rewards, dones, old_log_probs = self.evaluate()

        for _ in range(self.K):
            log_probs, dist_entropy = self.policy.evaluate(states, actions)

            importance_sampling, baseline, Z, len_counter = np.zeros_like(old_log_probs.detach()), 0, 0, 0
            for index, (p, q, reward, done) in enumerate(zip(log_probs.data.numpy()[::-1],
                                                             old_log_probs.data.numpy()[::-1],
                                                             rewards.data.numpy()[::-1],
                                                             dones[::-1])):
                ratio = np.exp(p - q)
                importance_sampling[len(importance_sampling) - index - 1] = ratio

                Z = reward + Z * self.gamma * (1 - done)
                len_counter = 1 + len_counter * (1 - done)
                baseline += ratio * Z / len_counter

            importance_sampling = T.tensor(importance_sampling)
            baseline /= importance_sampling.sum()

            loss_is = ((-log_probs * (rewards - baseline)) / importance_sampling).sum() / importance_sampling.sum()
            loss_entropy = (-0.0001 * dist_entropy).mean()
            loss = loss_is + loss_entropy

            losses.append(loss.item())

            # visualize
            # make_dot(loss, params=dict(self.policy.named_parameters())).render("attached")
            # raise SystemError

            self.policy.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
            self.policy.optimizer.step()

        self.update()
        return losses


def learn(env, agent, episodes=500):
    print('Episode: Mean Reward: Mean Loss: Mean Step')

    rewards = []
    losses = [0]
    steps = []
    num_episodes = episodes
    time_step = 0
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            time_step += 1
            action = agent.move(state)
            state_, reward, done, _ = env.step(action)
            agent.store((state, action, reward, done))

            state = state_
            total_reward += reward
            n_steps += 1

            if time_step % agent.update_interval == 0:
                loss = agent.learn()
                losses.extend(loss)

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
    agent = Agent(0.9, env.observation_space.shape, [env.action_space.n],
                  update_interval=100, K=2)

    learn(env, agent, 1000)
