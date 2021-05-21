import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import numpy as np
import gym

device = 'cuda:0' if T.cuda.is_available() else 'cpu'


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCriticNetwork, self).__init__()

        self.action_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.state_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def forward(self, state):
        action_probs = self.action_value(state)
        state_value = self.state_value(state)

        return action_probs, state_value


class AdvantageActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.9
        self.actorcritic = ActorCriticNetwork(state_dim, action_dim, 64)
        self.optimizer = optim.Adam(self.actorcritic.parameters(), lr=0.0005)

        self.state_memory = []
        self.action_memory = []
        self.logprob_memory = []
        self.reward_memory = []
        self.terminal_memory = []

    def move(self, state):
        state = T.from_numpy(state).float().to(device)
        action_probs, state_value = self.actorcritic(state)
        prob_dist = distributions.Categorical(action_probs)

        action = prob_dist.sample()
        log_prob = prob_dist.log_prob(action)

        self.logprob_memory.append(log_prob)
        self.state_memory.append(state)
        self.action_memory.append(action)

        return action.item()

    def evaluate(self, state, action):
        action_probs, state_value = self.actorcritic(state)
        dist = distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, T.squeeze(state_value), dist_entropy

    def store(self, reward, done):
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)

    def learn(self):
        rewards = []
        discounted_reward = 0
        for reward, terminal in zip(reversed(self.reward_memory), reversed(self.terminal_memory)):
            discounted_reward = 0 if terminal else reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = T.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        states = T.stack(self.state_memory).float().to(device)
        actions = T.stack(self.action_memory).float().to(device)
        logprobs, values, entropy = self.evaluate(states, actions)

        advantage = rewards - values
        actor_loss = -logprobs * advantage
        critic_loss = 0.5 * advantage ** 2
        entropy_loss = -0.001 * entropy

        loss = actor_loss + critic_loss + entropy_loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del self.state_memory[:]
        del self.action_memory[:]
        del self.logprob_memory[:]
        del self.reward_memory[:]
        del self.terminal_memory[:]

        return loss.item()


def learn(env, agent, episodes=500):
    print('Episode: 5 Episode Mean Reward: Last Loss: 5 Episode Mean Step'
          ' : Last Reward')

    rewards = []
    losses = []
    steps = []
    num_episodes = episodes
    for episode in range(num_episodes+1):
        done = False
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while not done:
            action = agent.move(state)
            state_, reward, done, _ = env.step(action)
            agent.store(reward, done)

            state = state_
            total_reward += reward
            n_steps += 1

        loss = agent.learn()
        rewards.append(total_reward)
        steps.append(n_steps)
        losses.append(loss)

        if episode % (episodes//10) == 0 and episode != 0:
            print(f'{episode:5d} : {np.mean(rewards[-5:]):5.2f} '
                  f': {losses[-1]: 5.2f}: {np.mean(steps[-5:]): 5.2f} '
                  f': {rewards[-1]: 3f}')

    return losses, rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = AdvantageActorCriticAgent(env.observation_space.shape[0], env.action_space.n)
    learn(env, agent, 500)