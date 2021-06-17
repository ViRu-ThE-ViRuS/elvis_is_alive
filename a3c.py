import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torchviz import make_dot

import gym
import numpy as np
from collections import deque

# {{{ SharedAdam


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
# }}}

# {{{ TransitionBuffer


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
# }}}

# {{{ ActorCriticNetwork


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActorCriticNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        self.actor = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.critic = nn.Linear(hidden_layer_dims[-1], 1)
        self.layers = nn.ModuleList(layers)

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))
        policy = F.softmax(self.actor(states), dim=-1)
        values = self.critic(states)

        return policy, values

    def move(self, states):
        policy, _ = self.forward(T.tensor(states).float())
        action = T.distributions.Categorical(policy).sample()
        return action.item()
# }}}

# {{{ Agent


class Agent(object):
    def __init__(self, gamma, env_function, max_episodes=5000, total_cpu=4, shared_optim=True):
        self.env_function = env_function
        self.gamma = gamma

        self.train_workers = total_cpu - 1
        self.num_episodes = max_episodes

        self.network_params = [128]
        self.episode_count, self.result_queue = mp.Value('i', 0), mp.Queue()

        env = env_function()
        self.model = ActorCriticNetwork(env.observation_space.shape, [env.action_space.n], self.network_params)
        self.optimizer = None
        self.model.share_memory()

        if shared_optim:
            self.optimizer = SharedAdam(self.model.parameters(), lr=0.009)

        self.trainers = [LocalTrainer(self.gamma, self.env_function, self.num_episodes,
                                      self.model, self.optimizer, self.network_params,
                                      self.episode_count, self.result_queue, i)
                         for i in range(self.train_workers)]

        self.tester = LocalTester(env, self.num_episodes, self.episode_count, self.result_queue, self.model)

    def run(self):
        [trainer.start() for trainer in self.trainers]
        self.tester.start()

        [trainer.join() for trainer in self.trainers]
        self.tester.join()
# }}}

# {{{ LocalTester


class LocalTester(mp.Process):
    def __init__(self, env, max_episodes, global_episode_count, global_result_queue, model):
        super(LocalTester, self).__init__()

        self.env = env
        self.max_episodes = max_episodes
        self.global_episodes = global_episode_count
        self.global_result_queue = global_result_queue
        self.global_model = model

        self.name = 'Tester'
        self.log_interval = 100

        self.rewards = []
        self.steps = []
        self.losses = [0]

        self.global_model.eval()

    def run(self):
        while self.global_episodes.value < self.max_episodes:
            for _ in range(self.log_interval):
                done = False
                state = self.env.reset()
                episode_reward = 0
                n_steps = 0

                while not done:
                    action = self.global_model.move(state)
                    state_, reward, done, _ = self.env.step(action)

                    state = state_
                    episode_reward += reward
                    n_steps += 1

                self.rewards.append(episode_reward)
                self.steps.append(n_steps)

            self.get_train_losses()

            print(f'{self.name}: {self.global_episodes.value:5d} : '
                  f'{np.mean(self.rewards[:-self.log_interval+1]):06.2f} : '
                  f'{np.mean(self.steps[:-self.log_interval+1]):06.2f} : '
                  f'{np.mean(self.losses):06.4f}')

        print(f'{self.name}: Training Complete with average({self.log_interval} episodes) '
              f'rewards: {np.mean(self.rewards[:-self.log_interval]):06.2f} :: '
              f'steps: {np.mean(self.steps[:-self.log_interval]):06.2f} :: '
              f'losses: {np.mean(self.losses): 06.4f}')

    def get_train_losses(self):
        self.losses = []

        for _ in range(self.log_interval):
            results = self.global_result_queue.get()

            if not results:
                break

            self.losses.append(results[-1])

# }}}

# {{{ LocalTrainer


class LocalTrainer(mp.Process):
    def __init__(self, gamma, env_function, max_episodes,
                 model, optimizer, network_params,
                 global_episodes, global_result_queue, name):
        super(LocalTrainer, self).__init__()

        self.gamma = gamma
        self.env = env_function()
        self.max_episodes = max_episodes

        self.global_episodes = global_episodes
        self.global_result_queue = global_result_queue
        self.name = 'Trainer#' + str(name)

        self.n_steps = 5
        self.critic_coeff = 0.5
        self.entropy_coeff = 0.001
        self.max_grad_norm = 0.5
        self.reward_scaling = True

        self.global_model = model
        self.optimizer = optimizer
        self.network_params = network_params

        if not self.optimizer:
            self.optimizer = T.optim.Adam(self.global_model.parameters(), lr=0.005, weight_decay=0.005)

        self.memory = TransitionBuffer(self.n_steps * 10)  # idk wadduh hek going on?
        self.local_model = ActorCriticNetwork(self.env.observation_space.shape,
                                              [self.env.action_space.n], self.network_params)

        self.pull_weights()

    def pull_weights(self):
        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.train()

    def run(self):
        while self.global_episodes.value < self.max_episodes:
            done = False
            state = self.env.reset()
            episode_reward = 0
            n_steps = 0

            while not done:
                for t in range(self.n_steps):
                    action = self.local_model.move(state)
                    state_, reward, done, _ = self.env.step(action)

                    self.memory.store((action, state, state_, reward, done))
                    state = state_

                    episode_reward += reward
                    n_steps += 1

                    if done:
                        break

            actor_loss, critic_loss, entropy_loss = self.calculate_loss()
            loss = (actor_loss + critic_loss + entropy_loss).mean()

            # visualize
            # make_dot(loss, params=dict(self.local_model.named_parameters())).render("attached")
            # raise SystemError

            self.push_weights(loss)
            self.global_result_queue.put([episode_reward, n_steps, loss.item()])

            with self.global_episodes.get_lock():
                self.global_episodes.value += 1

    def push_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.max_grad_norm)
        for local_param, global_param in zip(self.local_model.parameters(),
                                             self.global_model.parameters()):
            global_param._grad = local_param.grad

        self.optimizer.step()
        self.pull_weights()

    def get_discounted_reward(self, states_, rewards, R):
        state_values_ = T.squeeze(self.global_model(T.tensor(states_).float())[1].detach())

        discounted_rewards = np.zeros_like(rewards)
        for index, reward in enumerate(rewards[::-1]):
            discounted_rewards[len(rewards) - index - 1] = R = reward + self.gamma * R
        discounted_rewards = (self.gamma ** self.n_steps) * state_values_ + discounted_rewards

        if self.reward_scaling:
            discounted_rewards = ((discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5))

        return discounted_rewards

    def get_transition_values(self):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=True)

        states = T.tensor(states).float()
        actions = T.tensor(actions).float()
        action_probs, state_values = self.local_model(states)

        discounted_rewards = self.get_discounted_reward(states_, rewards,
                                                        0 if terminals[-1] else state_values[-1].item())

        dist = T.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return discounted_rewards, state_values, log_probs, dist_entropy

    def calculate_loss(self):
        rewards, state_values, log_probs, dist_entropy = self.get_transition_values()
        advantage = rewards - state_values

        actor_loss = - log_probs * advantage.detach()
        critic_loss = self.critic_coeff * advantage ** 2
        entropy_loss = - self.entropy_coeff * dist_entropy

        return actor_loss, critic_loss, entropy_loss
# }}}


def create_env():
    return gym.make('CartPole-v1')


if __name__ == '__main__':
    agent = Agent(0.9, create_env, max_episodes=3000, total_cpu=mp.cpu_count()-1, shared_optim=False)
    agent.run()
