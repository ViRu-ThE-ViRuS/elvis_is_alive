import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque

# {{{ TransitionMemory


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
# }}}

# {{{ ActorCriticNetwork


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActorCriticNetwork, self).__init__()

        actor_layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            actor_layers.append(nn.Linear(hidden_layer_dims[index], dim))
        actor_layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        critic_layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            critic_layers.append(nn.Linear(hidden_layer_dims[index], dim))
        critic_layers.append(nn.Linear(hidden_layer_dims[-1], 1))

        self.actor_layers = nn.ModuleList(actor_layers)
        self.critic_layers = nn.ModuleList(critic_layers)

        self.critic_loss = T.nn.MSELoss()

    def forward(self, states):
        x_actor = states.clone()
        for actor in self.actor_layers[:-1]:
            x_actor = T.tanh(actor(x_actor))
        policy = F.softmax(self.actor_layers[-1](x_actor), dim=-1)

        x_critic = states.clone()
        for critic in self.critic_layers[:-1]:
            x_critic = T.tanh(critic(x_critic))
        value = self.critic_layers[-1](x_critic)

        return policy, value

    def evaluate(self, states, actions):
        action_probs, state_values = self.forward(states)
        policy_dist = T.distributions.Categorical(action_probs)

        log_probs = policy_dist.log_prob(actions)
        dist_entropy = policy_dist.entropy()
        return log_probs, T.squeeze(state_values), dist_entropy
# }}}

# {{{ Agent


class Agent(object):
    def __init__(self, gamma, input_shape, output_shape,
                 update_interval=2000, epsilon_clip=0.2, K=10, c1=1.0, c2=0.01):
        self.gamma = gamma
        self.update_interval = update_interval
        self.epsilon_clip = epsilon_clip
        self.K = K
        self.c1 = c1
        self.c2 = c2
        self.learn_step = 0

        self.policy = ActorCriticNetwork(input_shape, output_shape, [64, 64])
        self.policy_old = ActorCriticNetwork(input_shape, output_shape, [64, 64])
        self.optimizer = T.optim.Adam(self.policy.parameters(), lr=0.001)
        self.memory = TransitionMemory(self.update_interval)

        self.update()

    def move(self, state):
        action_probs, _ = self.policy_old(T.tensor(state).float())
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def store(self, transition):
        self.memory.store(transition)

    def evaluate(self):
        (states, actions, states_, rewards, terminals) = self.memory.get_all(clear=True)

        states = T.tensor(states).float()
        actions = T.tensor(actions).float()

        old_log_probs, _, _ = self.policy_old.evaluate(states, actions)

        discounted_rewards, R = np.zeros_like(rewards), 0
        for index, (reward, done) in enumerate(zip(rewards[::-1], terminals[::-1])):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R * (1 - done)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        rewards = T.tensor(discounted_rewards).float()

        return states, actions, rewards, old_log_probs

    def update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

    def learn(self):
        states, actions, rewards, old_log_probs = self.evaluate()

        self.policy.train()
        self.learn_step += 1

        losses = []
        for _ in range(self.K):
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            importance_sampling = T.exp(log_probs - old_log_probs.detach())
            advantage = rewards - state_values.detach()

            _clip1 = importance_sampling * advantage
            _clip2 = T.clamp(importance_sampling, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage

            loss_clip = -T.min(_clip1, _clip2)
            loss_critic = self.c1 * self.policy.critic_loss(state_values, rewards)
            loss_entropy = -self.c2 * dist_entropy

            loss = (loss_clip + loss_critic + loss_entropy).mean()
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # visualize
            # make_dot(loss, params=dict(self.policy.named_parameters())).render("attached")
            # raise SystemError

        self.update()
        return losses


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

            if timestep % agent.update_interval == 0:
                loss = agent.learn()
                losses.extend(loss)

        rewards.append(total_reward)
        steps.append(n_steps)

        if episode % (episodes // 10) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards):06.2f} '
                  f': {np.mean(losses):06.4f} : {np.mean(steps):06.2f}')
            rewards = []
            # losses = [0]
            steps = []

    print(f'{episode:5d} : {np.mean(rewards):06.2f} '
          f': {np.mean(losses):06.4f} : {np.mean(steps):06.2f}')
    return losses, rewards
# }}}


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    agent = Agent(0.99, env.observation_space.shape, [env.action_space.n],
                  update_interval=2000, K=4, c1=1.0)

    learn(env, agent, 1000)
