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
        transitions = map(np.array, zip(*self.buffer))
        if clear:
            self.buffer.clear()

        return transitions

    def store(self, transition):
        self.buffer.append(transition)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims, lr):
        super(ActorCriticNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        for layer in layers:
            T.nn.init.orthogonal_(layer.weight.data)
            T.nn.init.zeros_(layer.bias.data)

        self.actor = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.critic = nn.Linear(hidden_layer_dims[-1], 1)

        T.nn.init.orthogonal_(self.actor.weight.data)
        T.nn.init.orthogonal_(self.critic.weight.data)
        T.nn.init.zeros_(self.actor.bias.data)
        T.nn.init.zeros_(self.critic.bias.data)

        self.layers = nn.ModuleList(layers)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, states):
        for layer in self.layers:
            states = T.relu(layer(states))
        action_probs = F.softmax(self.actor(states), dim=-1)
        state_values = self.critic(states)

        return action_probs, state_values

    def move(self, state):
        action_probs, _ = self.forward(T.tensor([state]).float())
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def evaluate_actions(self, states, actions=None):
        action_probs, state_values = self.forward(T.tensor(states).float())

        if actions is None:
            return state_values, None, None

        dist = T.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(T.tensor(actions).long())
        dist_entropy = dist.entropy()
        return state_values, log_probs, dist_entropy


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape,
                 n_steps=5, critic_coeff=0.5, entropy_coeff=0.02, max_grad_norm=0.5,
                 lr=0.001, gae_lambda=1.0, advantage_scaling=False, reward_scale=1,
                 network_params=[64]):
        self.gamma = gamma

        self.n_steps = n_steps
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.lr = lr
        self.gae_lambda = gae_lambda
        self.advantage_scaling = advantage_scaling
        self.reward_scale = reward_scale

        self.network_params = network_params
        self.policy = ActorCriticNetwork(input_shape, output_shape, network_params, lr=lr)
        self.memory = TransitionMemory(n_steps)

    def move(self, state):
        self.policy.eval()
        return self.policy.move(state)

    def store(self, transition):
        self.memory.store(transition)

    def calculate_advantages(self, states_, state_values, rewards, terminals):
        with T.no_grad():
            state_values = state_values.numpy().flatten()
            state_values_ = self.policy.evaluate_actions(states_)[0].numpy().flatten()

        lambda_return, advantages = 0.0, np.zeros_like(rewards)
        for index, (reward, terminal) in enumerate(zip(rewards[::-1], terminals[::-1])):
            inverse_index = len(rewards) - index - 1
            delta = reward + self.gamma * state_values_[inverse_index] * (1 - terminal) - state_values[inverse_index]
            lambda_return = delta + self.gamma * self.gae_lambda * (1 - terminal) * lambda_return
            advantages[inverse_index] = lambda_return

        rollout_rewards = advantages + state_values

        if self.advantage_scaling:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return T.tensor(rollout_rewards).float(), T.tensor(advantages).float()

    def learn(self):
        self.policy.train()

        actions, states, states_, rewards, terminals = self.memory.get_all(clear=True)
        state_values, log_probs, dist_entropy = self.policy.evaluate_actions(states, actions)
        rollout_rewards, advantages = self.calculate_advantages(states_, state_values, rewards, terminals)

        actor_loss = - log_probs * advantages
        critic_loss = self.critic_coeff * (rollout_rewards - state_values.flatten()) ** 2
        entropy_loss = - self.entropy_coeff * dist_entropy

        loss = (actor_loss + critic_loss + entropy_loss).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.policy.optimizer.step()

        # visualize
        # make_dot(loss, params=dict(self.policy.named_parameters())).render("attached")
        # raise SystemError

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
            for _ in range(agent.n_steps):
                action = agent.move(state)
                state_, reward, done, _ = env.step(action)
                agent.store((action, state, state_, reward, done))

                state = state_
                total_reward += reward
                n_steps += 1

                if done:
                    break

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
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLander-v2')
    agent = Agent(0.90, env.observation_space.shape, [env.action_space.n],
                  n_steps=5, entropy_coeff=0.0001, critic_coeff=0.5, lr=0.0007,
                  gae_lambda=1.0, network_params=[64, 64])

    learn(env, agent, 1000)
