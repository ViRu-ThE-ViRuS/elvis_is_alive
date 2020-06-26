import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer:
    def __init__(self, size, input_shape, n_outputs):
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.mem_size = size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_outputs))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, dims, n_actions):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.dims = dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*input_dims, dims[0])
        self.fc2 = nn.Linear(dims[0], dims[1])

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn1 = nn.LayerNorm(dims[0])
        self.bn2 = nn.LayerNorm(dims[1])

        self.action_value = nn.Linear(self.n_actions, dims[1])

        self.q = nn.Linear(dims[1], 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = T.optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.alpha = alpha
        self.input_dims = input_dims
        self.dims = dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*input_dims, dims[0])
        self.fc2 = nn.Linear(dims[0], dims[1])

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn1 = nn.LayerNorm(dims[0])
        self.bn2 = nn.LayerNorm(dims[1])

        self.mu = nn.Linear(dims[1], n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = T.tanh(self.mu(x))
        return x


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, n_actions,
                 gamma=0.99, mem_size=100000, dims=[300, 600], batch_size=64):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.input_dims = input_dims
        self.tau = tau

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, dims, n_actions)
        self.critic = CriticNetwork(beta, input_dims, dims, n_actions)

        self.target_actor = ActorNetwork(alpha, input_dims, dims, n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, dims, n_actions)

        self.noise = OUActionNoise(np.zeros(n_actions))
        self.update_network_params(tau=1)

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def remember(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def choose_action(self, observation):
        self.actor.eval()

        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation)  # .to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.sample(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        self.actor.eval()

        target_action = self.target_actor(state_)
        critic_value = self.critic(state, action)
        critic_value_ = self.target_critic(state_, target_action)

        # target = reward + self.gamma * critic_value_ * (1-done)
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * (1-done[j]))
        target = T.tensor(target).to(self.actor.device).view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        mu = self.actor(state)
        self.actor.train()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        actor_loss = - self.critic(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_params()

        # print(actor_loss.item(), critic_loss.item())


env = gym.make('LunarLanderContinuous-v2')
agent = Agent(0.000025, 0.00025, [8], 0.001, env, 2)

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    score_history.append(score)

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
