import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

import gym
import numpy as np
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(self, mem_size, alpha=0.6, beta_initial=0.4, beta_frames=500):
        self.mem_size = mem_size

        self.alpha = alpha
        self.beta_initial = beta_initial
        self.beta_frames = beta_frames
        self.frame_counter = 0

        self.buffer = deque(maxlen=mem_size)
        self.priorities = deque(maxlen=mem_size)

    def sample(self, batch_size, epsilon_link=None):
        sample_size = min(batch_size, len(self.buffer))
        sample_probs = self.get_probabilities()
        sample_indices = np.random.choice(len(self.buffer), sample_size, p=sample_probs)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices], epsilon_link)

        return map(list, zip(*samples)), importance, sample_indices

    def store(self, transition):
        self.buffer.append(transition)
        self.priorities.append(max(self.priorities, default=1.0))
        self.frame_counter += 1

    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probabilities = scaled_priorities / sum(scaled_priorities)
        return probabilities

    def get_importance(self, probabilities, epsilon_link=None):
        scaler = self.beta if not epsilon_link else epsilon_link
        importance = 1/len(self.buffer) * 1/probabilities ** scaler
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            self.priorities[index] = error + 1e-5

    @property
    def beta(self):
        return min(1.0, self.beta_initial + self.frame_counter * (1 - self.beta_initial) / self.beta_frames)


class DeepQN(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(DeepQN, self).__init__()

        layers = []
        layers.append(nn.Linear(*input_shape, hidden_layer_dims[0]))
        for index, dim in enumerate(hidden_layer_dims[1:]):
            layers.append(nn.Linear(hidden_layer_dims[index], dim))
        layers.append(nn.Linear(hidden_layer_dims[-1], *output_shape))

        self.layers = nn.ModuleList(layers)
        self.optimizer = T.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, states):
        for layer in self.layers[:-1]:
            states = F.relu(layer(states))
        return self.layers[-1](states)

    def learn(self, predictions, targets, importance):
        loss = F.mse_loss(input=predictions, target=targets, reduction='none') * importance
        errors = abs(loss.data.numpy())

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, errors


class Agent:
    def __init__(self, epsilon, gamma, input_shape, output_shape):
        self.epsilon = epsilon
        self.gamma = gamma
        self.output_shape = output_shape

        self.q_eval = DeepQN(input_shape, output_shape, [64, 64])
        self.q_target = DeepQN(input_shape, output_shape, [64, 64])
        self.memory = PrioritizedReplayBuffer(10000)

        self.learn_step = 0
        self.tau = 8
        self.batch_size = 32
        self.epsilon_link = False

        self.update()

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(*self.output_shape)
        else:
            self.q_eval.eval()
            state = T.tensor([state]).float()
            action = self.q_eval(state).max(axis=1)[1]
            return action.item()

    def update(self):
        if self.learn_step % self.tau == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            self.q_target.eval()

    def sample(self):
        (actions, states, states_, rewards, terminals), importance, indices = \
            self.memory.sample(self.batch_size, (1 - self.epsilon) if self.epsilon_link else None)

        actions = T.tensor(actions).long()
        states = T.tensor(states).float()
        states_ = T.tensor(states_).float()
        rewards = T.tensor(rewards).float()
        terminals = T.tensor(terminals).long()
        importance = T.tensor(importance).float()

        return (actions, states, states_, rewards, terminals), importance, indices

    def learn(self, state, action, state_, reward, done):
        self.learn_step += 1

        self.q_eval.train()
        self.memory.store((action, state, state_, reward, done))

        (actions, states, states_, rewards, terminals), importance, sample_indices = self.sample()

        indices = np.arange(len(actions))
        q_eval = self.q_eval(states)[indices, actions]
        target_actions = self.q_eval(states_).detach().max(axis=1)[1]
        q_target = self.q_target(states_).detach()[indices, target_actions]
        q_target = rewards + self.gamma * q_target * (1 - terminals)

        loss, errors = self.q_eval.learn(q_eval, q_target, importance)
        self.epsilon *= 0.99 if self.epsilon > 0.05 else 1.0

        self.memory.set_priorities(sample_indices, errors)

        # visualize
        # make_dot(loss, params=dict(self.q_eval.named_parameters())).render("attached")

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
            loss = agent.learn(state, action, state_, reward, done)

            state = state_
            total_reward += reward
            n_steps += 1

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
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    agent = Agent(1.0, 0.9, env.observation_space.shape, [env.action_space.n])

    learn(env, agent, 500)
