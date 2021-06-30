import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import numpy as np
import gym
from collections import deque


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


class ActionValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActionValueNetwork, self).__init__()

        q1 = [nn.Linear(input_shape[0] + output_shape[0], hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            q1.append(nn.Linear(hidden_layer_dims[index], dim))
        q1.append(nn.Linear(hidden_layer_dims[-1], 1))

        q2 = [nn.Linear(input_shape[0] + output_shape[0], hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            q2.append(nn.Linear(hidden_layer_dims[index], dim))
        q2.append(nn.Linear(hidden_layer_dims[-1], 1))

        self.q1 = nn.ModuleList(q1)
        self.q2 = nn.ModuleList(q2)

    def forward(self, states, actions):
        x1 = T.cat([states, actions], dim=1)
        x2 = x1.clone().detach()

        for q1, q2 in zip(self.q1[:-1], self.q2[:-1]):
            x1 = T.relu(q1(x1))
            x2 = T.relu(q2(x2))

        x1, x2 = self.q1[-1](x1), self.q2[-1](x2)
        return x1, x2


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims, action_space=None):
        super(GaussianPolicyNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        self.layers = nn.ModuleList(layers)
        self.mean = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.log_std = nn.Linear(hidden_layer_dims[-1], *output_shape)

        # action scaling
        if action_space is None:
            self.action_scale = T.tensor(1.0).float()
            self.action_bias = T.tensor(0.0).float()
        else:
            self.action_scale = T.tensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = T.tensor((action_space.high + action_space.low) / 2.0)

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))

        mean = self.mean(states)
        log_std = T.clamp(self.log_std(states), -20, 2)
        return mean, log_std

    def sample(self, states):
        mean, log_std = self.forward(states)
        mean = T.tanh(mean) * self.action_scale + self.action_bias

        std = log_std.exp()
        normal = T.distributions.Normal(mean, std)

        x_t = normal.rsample()  # reparam trick: grad wrt mean and std, sample eps from std normal
        y_t = T.tanh(x_t)       # squash to bounded values
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= T.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # scale log prob
        log_prob = log_prob.sum(1, keepdim=True)  # log prob of each action-space value

        return action, log_prob, mean


class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims, action_space=None):
        super(DeterministicPolicyNetwork, self).__init__()

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))

        self.layers = nn.ModuleList(layers)
        self.mean = nn.Linear(hidden_layer_dims[-1], *output_shape)
        self.noise = T.tensor(*output_shape)

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = T.Tensor((action_space.high - action_space.low) / 2.0).float()
            self.action_bias = T.Tensor((action_space.high + action_space.low) / 2.0).float()

    def forward(self, states):
        for layer in self.layers:
            states = F.relu(layer(states))

        mean = self.mean(states)
        return T.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, states):
        mean = self.forward(states)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, T.tensor(0.0), mean


class Agent:
    def __init__(self, input_shape, output_shape, args):
        self.alpha = args.get('alpha', 0.2)
        self.gamma = args.get('gamma', 0.99)
        self.tau = args.get('tau', 0.005)
        self.K = args.get('K', 1)
        self.lr = args.get('lr', 0.0001)
        self.batch_size = args.get('batch_size', 256)
        self.target_update_interval = args.get('target_update_interval', 1)
        self.automatic_entropy_tuning = args.get('automatic_entropy_tuning', False)
        self.critic_nn_shape = args.get('critic_nn_shape', [64])
        self.actor_nn_shape = args.get('actor_nn_shape', [64])

        action_space = args.get('action_space', None)

        self.critic = ActionValueNetwork(input_shape, output_shape, self.critic_nn_shape)
        self.critic_ = ActionValueNetwork(input_shape, output_shape, self.critic_nn_shape)
        self.update(soft=False)

        if args.get('policy_type', 'gaussian') == 'gaussian':
            if self.automatic_entropy_tuning:
                self.target_entropy = -T.prod(T.tensor(action_space.shape)).item()
                self.log_alpha = T.zeros(1, requires_grad=True)
                self.alpha_optim = T.optim.Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicyNetwork(input_shape, output_shape, self.actor_nn_shape, action_space)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicyNetwork(input_shape, output_shape, self.actor_nn_shape, action_space)

        self.critic_optim = T.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.policy_optim = T.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(self.batch_size * 10000)
        self.learn_step = 0

    def update(self, soft=True):
        if soft:
            for target_param, param in zip(self.critic_.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                        + param.data * self.tau)

        else:
            self.critic_.load_state_dict(self.critic.state_dict())

    def move(self, state, eval=False):
        state = T.tensor(state).float().unsqueeze(0)

        if eval:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)

        return action.detach().numpy()[0]

    def store(self, transition):
        self.memory.store(transition)

    def sample_memory(self):
        states, actions, states_, rewards, terminals = self.memory.sample(self.batch_size)

        states = T.tensor(states).float()
        actions = T.tensor(actions).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float().unsqueeze(1)
        terminals = T.tensor(terminals).long().unsqueeze(1)

        return states, actions, states_, rewards, terminals

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.learn_step += 1
        states, actions, states_, rewards, terminals = self.sample_memory()

        with T.no_grad():
            actions_, log_pi_, _ = self.policy.sample(states_)
            q1_next, q2_next = self.critic_(states_, actions_)
            q_next = T.min(q1_next, q2_next) - self.alpha * log_pi_
            q_targets = rewards + self.gamma * q_next * (1 - terminals)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        new_actions, log_pi, _ = self.policy.sample(states)
        q_actions = T.min(*self.critic(states, new_actions))
        policy_loss = ((self.alpha * log_pi) - q_actions).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = T.tensor(0.0)

        if self.learn_step % self.target_update_interval == 0:
            self.update(soft=True)

        # visualize
        # make_dot(critic_loss, params=dict(self.critic.named_parameters())).render("attached")
        # raise SystemError

        return (critic_loss + policy_loss + alpha_loss).item()


def learn(env, agent, args):
    episodes = args['episodes']
    start_steps = args['start_steps']

    print('Episode: Mean Reward: Mean Loss: Mean Step')

    rewards = []
    losses = [0]
    steps = []
    num_episodes = episodes
    total_steps = 0
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.move(state)

            state_, reward, done, _ = env.step(action)
            agent.store((state, action, state_, reward, done))

            state = state_
            total_reward += reward
            n_steps += 1
            total_steps += 1

            for _ in range(agent.K):
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
    env = gym.make('LunarLanderContinuous-v2')

    agent_args = {
        'alpha': 0.2,
        'gamma': 0.99,
        'tau': 0.005,
        'K': 1,
        'lr': 0.0001,
        'batch_size': 256,
        'target_update_interval': 1,
        'automatic_entropy_tuning': False,
        'action_space': None,
        'policy': 'gaussian'
    }
    agent = Agent(env.observation_space.shape, [env.action_space.shape[0]], agent_args)

    run_args = {
        'episodes': 100,
        'start_steps': 5000
    }
    learn(env, agent, run_args)
