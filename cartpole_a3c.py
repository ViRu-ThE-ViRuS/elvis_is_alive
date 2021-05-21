import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.distributions import Categorical
import torch.multiprocessing as mp

cpu_count = mp.cpu_count()
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 300
max_test_ep = 400

state_dim = 4
action_dim = 2


def create_env():
    return gym.make('CartPole-v1')


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)

        return action_probs, state_values

    def move(self, state):
        state = T.from_numpy(state).float()
        action_probs, _ = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item()


def train(shared_model, rank):
    local_model = ActorCritic(state_dim, action_dim)
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = optim.Adam(shared_model.parameters(), lr=learning_rate)
    env = create_env()

    for episode in range(max_test_ep):
        done = False
        state = env.reset()

        while not done:
            state_memory, action_memory, reward_memory = [], [], []

            for t in range(update_interval):
                action = local_model.move(state)
                state_, reward, done, _ = env.step(action)

                state_memory.append(state)
                reward_memory.append(reward/100)
                action_memory.append(action)

                state = state_

                if done:
                    break

            final_state_value = local_model(T.tensor(state).float())[1].item()
            R = 0.0 if done else final_state_value

            td_targets = []
            for reward in reward_memory[::-1]:
                R = reward + gamma * R
                td_targets.append([R])
            td_targets.reverse()

            states = T.tensor(state_memory).float()
            actions = T.tensor(action_memory).view(-1, 1)
            td_targets = T.tensor(td_targets)

            action_probs, state_values = local_model(states)
            entropy = Categorical(action_probs).entropy()
            advantage = td_targets - state_values

            action_probs = action_probs.gather(1, actions)

            actor_loss = - T.log(action_probs) * advantage.detach()
            critic_loss = F.smooth_l1_loss(state_values, td_targets.detach())
            entropy_loss = - entropy
            loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy_loss

            optimizer.zero_grad()
            loss.mean().backward()

            for shared_param, local_param in zip(shared_model.parameters(),
                                                 local_model.parameters()):
                shared_param._grad = local_param.grad

            optimizer.step()
            local_model.load_state_dict(shared_model.state_dict())

    env.close()
    print('Worker: {} training complete'.format(rank))


def test(shared_model):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep+1):
        done = False
        state = env.reset()
        while not done:
            action = shared_model.move(state)
            state_, reward, done, _ = env.step(action)
            state = state_
            score += reward

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()


if __name__ == '__main__':
    global_model = ActorCritic(state_dim, action_dim)
    global_model.share_memory()

    processes = []
    for rank in range(cpu_count + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model, ))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
