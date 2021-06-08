import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


MAX_EP = 5000


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

        self.actor = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.critic = nn.Linear(hidden_layer_dims[-1], 1)

        self.layers = nn.ModuleList(layers)

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))
        pi = F.softmax(self.actor(states), dim=0)
        v = self.critic(states)

        return pi, v


class Agent(object):
    def __init__(self, gamma, env_name):
        self.env_name = env_name
        self.gamma = gamma

        env = gym.make(env_name)
        self.actor_critic = ActorCriticNetwork(env.observation_space.shape,
                                               [env.action_space.n], [64, 64])
        self.actor_critic.share_memory()

        self.optimizer = T.optim.Adam(self.actor_critic.parameters(), lr=0.005)
        self.episode, self.reward, self.result_queue = mp.Value('i', 0), mp.Value('d', 0), mp.Queue()

        self.workers = [LocalAgent(gamma, env_name,
                                   self.actor_critic, self.optimizer, self.episode, self.reward, self.result_queue, i)
                        for i in range(mp.cpu_count())]

    def train(self):
        [worker.start() for worker in self.workers]

        results = []
        while True:
            try:
                result = self.result_queue.get(True, 5)
                results.append(result)
            except Exception:
                break

        [worker.join() for worker in self.workers]
        return results


class LocalAgent(mp.Process):
    def __init__(self, gamma, env_name,
                 model, optimizer, global_episodes, global_rewards, global_result_queue, name):
        super(LocalAgent, self).__init__()

        self.global_model = model
        self.global_optimizer = optimizer
        self.global_episodes = global_episodes
        self.global_rewards = global_rewards
        self.global_result_queue = global_result_queue
        self.name = str(name)

        self.gamma = gamma
        self.env = gym.make(env_name)
        self.actor_critic = ActorCriticNetwork(self.env.observation_space.shape,
                                               [self.env.action_space.n], [64, 64])
        self.memory = TransitionMemory(1000)

    def run(self):
        while self.global_episodes.value < MAX_EP:
            done = False
            state = self.env.reset()
            total_reward = 0
            n_steps = 0

            while not done:
                action = self.move(state)
                state_, reward, done, _ = self.env.step(action)
                self.store((action, state, state_, reward, done))

                state = state_
                total_reward += reward
                n_steps += 1

            loss = self.push_pull()
            self.record(total_reward, n_steps, loss)

    def move(self, state):
        self.actor_critic.eval()
        action_probs, _ = self.actor_critic(T.tensor(state, dtype=T.float))
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def store(self, transition):
        self.memory.store(transition)

    def push_pull(self):
        loss = self.learn()

        self.global_optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(self.actor_critic.parameters(),
                                             self.global_model.parameters()):
            global_param._grad = local_param.grad

        self.global_optimizer.step()
        self.actor_critic.load_state_dict(self.global_model.state_dict())

        return loss.item()

    def record(self, total_reward, steps, loss):
        with self.global_episodes.get_lock():
            self.global_episodes.value += 1

        with self.global_rewards.get_lock():
            if self.global_rewards.value == 0:
                self.global_rewards.value = total_reward
            else:
                self.global_rewards.value = self.global_rewards.value * 0.99 + total_reward * 0.01

        self.global_result_queue.put(self.global_rewards.value)

        if self.global_episodes.value % (self.global_episodes.value // 10) == 0:
            print(f'{self.global_episodes.value:5d} : {total_reward:06.2f} '
                  f': {loss:06.4f} : {steps:06.2f}')

    def get_all(self, clear=True):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=clear)

        discounted_rewards, _r = np.zeros_like(rewards), 0
        for index, (reward, terminal) in enumerate(zip(reversed(rewards), reversed(terminals))):
            discounted_rewards[len(rewards) - index - 1] = _r = reward * (1 - terminal) + self.gamma * _r
        rewards = T.tensor(discounted_rewards).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        terminals = T.tensor(terminals).long

        return actions, states, states_, rewards, terminals

    def evaluate(self, state, action):
        action_probs, state_value = self.actor_critic(state)
        dist = T.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return log_probs, T.squeeze(state_value), dist_entropy

    def learn(self):
        self.actor_critic.train()
        actions, states, states_, rewards, terminals = self.get_all(clear=True)

        try:
            log_probs, state_values, dist_entropy = self.evaluate(states, actions)
        except Exception:
            print(states)
            raise SystemError

        advantage = rewards - state_values

        actor_loss = -log_probs * advantage
        critic_loss = advantage ** 2
        entropy_loss = dist_entropy
        loss = (actor_loss + critic_loss + entropy_loss).mean()

        # visualize
        # make_dot(loss, params=dict(self.actor_critic.named_parameters())).render("attached")

        return loss


if __name__ == '__main__':
    agent = Agent(0.9, 'CartPole-v1')
    agent.train()
