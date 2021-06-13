import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchviz import make_dot

import numpy as np
import gym
from collections import deque
import time


class SharedAdam(T.optim.Adam):
    def __init__(self, *args, **kwargs):
        super(SharedAdam, self).__init__(*args, **kwargs)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(param.data)
                state['exp_avg_sq'] = T.zeros_like(param.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class TransitionBuffer:
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)

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
    def __init__(self, gamma, env_name, max_episodes=500, total_cpu=4):
        self.env_name = env_name
        self.gamma = gamma

        self.cpu_count_test = 1
        self.cpu_count_train = total_cpu - 1
        self.max_episodes_train = max_episodes * total_cpu
        self.max_episodes_test = max_episodes

        self.train_episode, self.test_episode, self.result_queue = mp.Value('i', 0), mp.Value('i', 0),  mp.Queue()

        env = gym.make(env_name)
        self.global_model = ActorCriticNetwork(env.observation_space.shape, [env.action_space.n], [128])
        self.global_model.share_memory()
        self.optimizer = SharedAdam(self.global_model.parameters(), lr=0.001)

        self.trainers = [LocalTrainer(self.gamma, self.env_name, self.max_episodes_train,
                                      self.global_model, self.optimizer,
                                      self.train_episode, self.result_queue, i)
                         for i in range(self.cpu_count_train)]

        self.testers = [LocalTester(self.env_name, self.max_episodes_test,
                                    self.global_model, self.test_episode, self.result_queue, 0)]

    def run(self):
        [trainer.start() for trainer in self.trainers]
        [tester.start() for tester in self.testers]

        results = []
        while True:
            try:
                result = self.result_queue.get(True, 5)
                results.append(result)
            except Exception:
                break

        [trainer.join() for trainer in self.trainers]
        [tester.join() for tester in self.testers]
        return results


class LocalTester(mp.Process):
    def __init__(self, env_name, max_episodes,
                 model, global_episodes, global_result_queue, name):
        super(LocalTester, self).__init__()

        self.global_model = model
        self.global_episodes = global_episodes
        self.global_result_queue = global_result_queue
        self.max_episodes = max_episodes
        self.name = 'Tester#' + str(name)

        self.env = gym.make(env_name)

    def run(self):
        while self.global_episodes.value < self.max_episodes:
            done = False
            state = self.env.reset()
            total_reward = 0
            n_steps = 0

            while not done:
                action = self.move(state)
                state_, reward, done, _ = self.env.step(action)

                state = state_
                total_reward += reward
                n_steps += 1

            self.record(total_reward, n_steps)

    def move(self, state):
        self.global_model.eval()
        action_probs, _ = self.global_model(T.tensor(state, dtype=T.float))
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def record(self, total_reward, steps):
        with self.global_episodes.get_lock():
            self.global_episodes.value += 1

        self.global_result_queue.put(total_reward)
        if self.global_episodes.value % (self.max_episodes // 10) == 0:
            print(f'{self.name}: {self.global_episodes.value:5d} : {total_reward:06.2f} : {steps:06.2f}')
            time.sleep(2.5)


class LocalTrainer(mp.Process):
    def __init__(self, gamma, env_name, max_episodes,
                 model, optimizer, global_episodes, global_result_queue, name):
        super(LocalTrainer, self).__init__()

        self.global_model = model
        self.global_optimizer = optimizer
        self.global_episodes = global_episodes
        self.global_result_queue = global_result_queue
        self.max_episodes = max_episodes
        self.name = 'Trainer#' + str(name)

        self.gamma = gamma
        self.env = gym.make(env_name)
        self.memory = TransitionBuffer()

        self.actor_critic = ActorCriticNetwork(self.env.observation_space.shape, [self.env.action_space.n], [128])
        self.sync()

    def sync(self):
        self.actor_critic.load_state_dict(self.global_model.state_dict())

    def run(self):
        while self.global_episodes.value < self.max_episodes:
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

            loss = self.push_grads()
            self.record(total_reward, n_steps, loss)

    def push_grads(self):
        loss = self.calculate_loss()

        self.global_optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(self.actor_critic.parameters(),
                                             self.global_model.parameters()):
            global_param._grad = local_param.grad

        self.global_optimizer.step()
        self.sync()

        return loss.item()

    def record(self, total_reward, steps, loss):
        with self.global_episodes.get_lock():
            self.global_episodes.value += 1

        self.global_result_queue.put(total_reward)
        if self.global_episodes.value % (self.max_episodes // 10) == 0:
            # print(f'{self.global_episodes.value:5d} : {total_reward:06.2f} '
            #       f': {loss:06.4f} : {steps:06.2f}')
            pass

    def store(self, transition):
        self.memory.store(transition)

    def evaluate(self, clear=True):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=clear)
        log_probs, state_values, state_values_, dist_entropy = self._evaluate(states, states_, actions)

        discounted_rewards, R = np.zeros_like(rewards), 0 if terminals[-1] else state_values[-1].item()
        for index, (reward, done) in enumerate(zip(rewards[::-1], terminals[::-1])):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R * (1 - done)
        rewards = T.tensor(discounted_rewards).float()

        rewards = rewards + self.gamma * state_values_
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        return log_probs, rewards.detach(), state_values, dist_entropy

    def _evaluate(self, states, states_, actions):
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        actions = T.tensor(actions).float()

        action_probs, state_values = self.actor_critic(states)
        dist = T.distributions.Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        _, state_values_ = self.global_model(states_)
        return log_probs, T.squeeze(state_values), T.squeeze(state_values_.detach()), dist_entropy

    def move(self, state):
        self.actor_critic.eval()
        action_probs, _ = self.actor_critic(T.tensor(state, dtype=T.float))
        action = T.distributions.Categorical(action_probs).sample()
        return action.item()

    def calculate_loss(self):
        self.actor_critic.train()

        log_probs, rewards, state_values, dist_entropy = self.evaluate()
        advantage = rewards - state_values

        actor_loss = -log_probs * advantage.detach()
        critic_loss = advantage ** 2
        entropy_loss = -dist_entropy * 0.001
        loss = (actor_loss + critic_loss + entropy_loss).mean()

        # visualize
        # make_dot(loss, params=dict(self.actor_critic.named_parameters())).render("attached")

        return loss


if __name__ == '__main__':
    agent = Agent(0.9, 'CartPole-v1', 300, mp.cpu_count())
    agent.run()
