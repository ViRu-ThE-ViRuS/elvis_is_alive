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
        transitions = map(np.array, zip(*self.buffer))
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
        self._init_layers()

    def _init_layers(self):
        for layer in self.layers:
            T.nn.init.orthogonal_(layer.weight.data)
            T.nn.init.zeros_(layer.bias.data)

        T.nn.init.orthogonal_(self.actor.weight.data)
        T.nn.init.orthogonal_(self.critic.weight.data)
        T.nn.init.zeros_(self.actor.bias.data)
        T.nn.init.zeros_(self.critic.bias.data)

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
# }}}

# {{{ Agent


class Agent(object):
    def __init__(self, gamma, env_function, max_episodes=5000, total_cpu=4, shared_optim=True,
                 network_params=[128], lr=0.005):
        self.env_function = env_function
        self.gamma = gamma

        self.train_workers = total_cpu - 1
        self.num_episodes = max_episodes

        self.network_params = network_params
        self.lr = lr
        self.episode_count, self.result_queue = mp.Value('i', 0), mp.Queue()

        env = env_function()
        self.model = ActorCriticNetwork(env.observation_space.shape, [env.action_space.n], self.network_params)
        self.model.share_memory()
        self.optimizer = SharedAdam(self.model.parameters(), lr=self.lr) if shared_optim else None

        self.trainers = [LocalTrainer(self.gamma, self.env_function, self.num_episodes,
                                      self.model, self.optimizer, self.network_params,
                                      self.episode_count, self.result_queue, i,
                                      self.lr)
                         for i in range(self.train_workers)]

        self.tester = LocalTester(env, self.num_episodes, self.episode_count, self.result_queue, self.model)

    def run(self):
        self.tester.start()
        [trainer.start() for trainer in self.trainers]

        [trainer.join() for trainer in self.trainers]
        self.tester.join(timeout=1.0)
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

        print(f'Training Complete with average({self.log_interval} episodes) '
              f'rewards: {np.mean(self.rewards[:-self.log_interval]):06.2f} :: '
              f'steps: {np.mean(self.steps[:-self.log_interval]):06.2f} :: '
              f'losses: {np.mean(self.losses):06.4f}')

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
                 global_episodes, global_result_queue, name,
                 lr, n_steps=5, critic_coeff=0.05, entropy_coeff=0.001,
                 max_grad_norm=0.5, gae_lambda=1.0, advantage_scaling=False):
        super(LocalTrainer, self).__init__()

        self.gamma = gamma
        self.env = env_function()
        self.max_episodes = max_episodes

        self.global_episodes = global_episodes
        self.global_result_queue = global_result_queue
        self.name = 'Trainer#' + str(name)

        self.n_steps = n_steps
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.lr = lr
        self.gae_lambda = gae_lambda
        self.advantage_scaling = advantage_scaling

        self.global_model = model
        self.optimizer = optimizer
        self.network_params = network_params

        if not self.optimizer:
            self.optimizer = T.optim.Adam(self.global_model.parameters(), lr=self.lr)

        self.memory = TransitionBuffer(self.n_steps)
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
            losses = []

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

                self.push_weights(loss)
                losses.append(loss.item())

            # visualize
            # make_dot(loss, params=dict(self.local_model.named_parameters())).render("attached")
            # raise SystemError

            self.global_result_queue.put([episode_reward, n_steps, np.mean(losses)])

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

    def calculate_advantages(self, states_, state_values, rewards, terminals):
        with T.no_grad():
            state_values = state_values.numpy().flatten()
            state_values_ = self.local_model.evaluate_actions(states_)[0].numpy().flatten()

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

    def calculate_loss(self):
        actions, states, states_, rewards, terminals = self.memory.get_all(clear=True)
        state_values, log_probs, dist_entropy = self.local_model.evaluate_actions(states, actions)
        rollout_rewards, advantages = self.calculate_advantages(states_, state_values, rewards, terminals)

        actor_loss = - log_probs * advantages
        critic_loss = self.critic_coeff * (rollout_rewards - state_values.flatten()) ** 2
        entropy_loss = - self.entropy_coeff * dist_entropy

        return actor_loss, critic_loss, entropy_loss
# }}}


def create_env():
    return gym.make('CartPole-v1')


if __name__ == '__main__':
    agent = Agent(0.90, create_env, max_episodes=3000, total_cpu=mp.cpu_count() - 1,
                  shared_optim=True, lr=0.0005)
    agent.run()
