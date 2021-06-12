import torch as T
import torch.nn as nn
from torch.distributions import Categorical
from torchviz import make_dot
import gym

device = 'cuda:0' if T.cuda.is_available() else 'cpu'


class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = T.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)
        return action_logprobs, T.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = T.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()

    def update(self, memory):
        # mc estimate of state reward, and normalizing
        rewards = []
        discounted_reward = 0
        for reward, terminal in zip(reversed(memory.rewards), reversed(memory.terminals)):
            discounted_reward = 0 if terminal else reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = T.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = T.stack(memory.states).to(device).detach()
        old_actions = T.stack(memory.actions).to(device).detach()
        old_logprobs = T.stack(memory.logprobs).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = T.exp(logprobs - old_logprobs.detach())  # pi(at|st) / pi_old(at|st)
            advantages = rewards - state_values.detach()

            l_clip_1 = ratios * advantages
            l_clip_2 = T.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss_clip = -T.min(l_clip_1, l_clip_2)
            loss_vf = 0.5 * self.loss(state_values, rewards)
            loss_entropy = -0.01 * dist_entropy  # encourage exploration

            loss = (loss_clip + loss_vf + loss_entropy).mean()

            # visualize
            # make_dot(loss, params=dict(self.policy.named_parameters())).render("attached")
            # raise SystemError

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = 4

    interval = 20
    max_episodes = 50000
    max_timesteps = 300
    hidden_dim = 64
    update_timestep = 2000

    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    memory = ReplayBuffer()
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip)

    total_reward = 0
    avg_length = 0
    timestep = 0

    for episode in range(max_episodes+1):
        state = env.reset()

        for t in range(max_timesteps):
            timestep += 1
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.terminals.append(done)

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear()

            total_reward += reward
            if done:
                break

        avg_length += t

        if episode % interval == 0:
            avg_length = int(avg_length / interval)
            total_reward = int(total_reward / interval)

            print(f'episode {episode:5d} : avg_length {avg_length:5d} : reward {total_reward:5d}')
            total_reward = avg_length = 0


main()
