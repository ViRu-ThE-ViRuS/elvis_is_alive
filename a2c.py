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
    def __init__(self, input_shape, output_shape, hidden_layer_dims, lr=0.001):
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
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))
        pi = F.softmax(self.actor(states), dim=0)
        v = self.critic(states)

        return pi, v


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape,
                 n_steps=5, critic_coeff=0.5, entropy_coeff=0.02, max_grad_norm=0.5,
                 lr=0.001):
        self.gamma = gamma
        self.policy = ActorCriticNetwork(input_shape, output_shape, [64, 64], lr=lr)
        self.policy_old = ActorCriticNetwork(input_shape, output_shape, [64, 64])
        self.memory = TransitionMemory(1000)

        self.n_steps = n_steps
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.update()

    def update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

    def move(self, state):
        action_probs, _ = self.policy_old(T.tensor(state, dtype=T.float))
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def store(self, transition):
        self.memory.store(transition)

    def evaluate(self, clear=True):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=clear)
        log_probs, state_values, state_values_, dist_entropy = self._evaluate(states, states_, actions)
        state_values_ = state_values_.detach()

        rewards, terminals = np.array(rewards), np.array(terminals)

        discounted_rewards, R = np.zeros_like(rewards), 0
        for index, (reward, done) in enumerate(zip(rewards[::-1], terminals[::-1])):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R * (1 - done)

        n_step_rewards = np.zeros_like(rewards)
        for index in range(len(rewards)):
            n_step_rollout = discounted_rewards[index:(min(len(rewards), index+self.n_steps))]
            next_done = np.where(terminals[index:min(len(rewards), index + self.n_steps)] == 1)[0]

            if len(next_done) != 0:
                n_step_rollout = n_step_rollout[:next_done[0]+1]
            n_step_rewards[index] = n_step_rollout.sum() + (self.gamma ** self.n_steps * state_values_[index])

        rewards = (n_step_rewards - n_step_rewards.mean()) / (n_step_rewards.std() + 1e-5)
        rewards = T.tensor(rewards).float()

        return log_probs, rewards, state_values, state_values_, dist_entropy

    def _evaluate(self, states, states_, actions):
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        actions = T.tensor(actions).float()

        self.policy.eval()
        action_probs, state_values = self.policy(states)
        _, state_values_ = self.policy_old(states_)

        dist = T.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return log_probs, T.squeeze(state_values), T.squeeze(state_values_), dist_entropy

    def learn(self):
        self.policy.train()
        log_probs, rewards, state_values, state_values_, dist_entropy = self.evaluate()
        advantage = rewards - state_values

        actor_loss = -log_probs * advantage.detach()
        critic_loss = self.critic_coeff * advantage ** 2
        entropy_loss = -self.entropy_coeff * dist_entropy
        loss = (actor_loss + critic_loss + entropy_loss).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.policy.optimizer.step()

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
    agent = Agent(0.99, env.observation_space.shape, [env.action_space.n],
                  n_steps=10, entropy_coeff=0.001, critic_coeff=0.5, lr=0.001)

    learn(env, agent, 1000)
