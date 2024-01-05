import sys
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import copy
import random


class Memory:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__memory = []
        self.__position = 0

    def push(self, element):
        if len(self.__memory) < self.__capacity:
            self.__memory.append(None)
        self.__memory[self.__position] = element
        self.__position = (self.__position + 1) % self.__capacity

    def sample(self, batch_size):
        return list(zip(*random.sample(self.__memory, batch_size)))

    def __len__(self):
        return len(self.__memory)


class MountainCarAgent:
    def __init__(self):
        self.__env = gym.make("MountainCar-v0")
        self.__device = torch.device("cuda")
        self.__target_update = 1000
        self.__batch_size = 128
        self.__max_steps = 100001
        self.__max_epsilon = 0.5
        self.__min_epsilon = 0.1
        self.__gamma = 0.99
        self.__model = None
        self.__target_model = None
        self.__optimizer = None

    def get_model(self):
        return self.__model

    def load(self, model_weights_path):
        self.__model.load_state_dict(torch.load(model_weights_path))
        self.__model.eval()

    def create_new_model(self):
        self.__model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.__target_model = copy.deepcopy(self.__model)
        self.__model.to(self.__device)
        self.__target_model.to(self.__device)
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.00003)

    def fit(self, reward, new_state, memory):
        batch = memory.sample(self.__batch_size)
        # Adjust the reward by adding a bonus based on the absolute value of the new_state's second component
        modified_reward = reward + 10 * abs(new_state[1])
        state, action, reward, next_state, done = batch
        state = torch.tensor(state).to(self.__device).float()
        next_state = torch.tensor(next_state).to(self.__device).float()
        reward = torch.tensor(reward).to(self.__device).float()
        action = torch.tensor(action).to(self.__device)
        done = torch.tensor(done).to(self.__device)
        memory.push((state, action, modified_reward, new_state, done))
        target_q = torch.zeros(reward.size()[0]).float().to(self.__device)
        with torch.no_grad():
            target_q = self.__target_model(next_state).max(1)[0].view(-1)
            target_q[done] = 0
        target_q = reward + target_q * self.__gamma

        q = self.__model(state).gather(1, action.unsqueeze(1))
        loss = F.mse_loss(q, target_q.unsqueeze(1))

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

    def select_action(self, state, epsilon, model):
        if random.random() < epsilon:
            return random.randint(0, 2)
        return model(torch.tensor(state).to(self.__device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def train(self):
        memory = Memory(5000)
        self.create_new_model()
        state = self.__env.reset()
        rewards_by_target_updates = []
        for step in range(self.__max_steps):
            epsilon = self.__max_epsilon - (self.__max_epsilon - self.__min_epsilon) * step / self.__max_steps
            action = self.select_action(state, epsilon, self.__model)
            new_state, reward, done, _ = self.__env.step(action)
            memory.push((state, action, reward, new_state, done))
            if done:
                state = self.__env.reset()
                done = False
            else:
                state = new_state
            if step > self.__batch_size:
                self.fit(reward, new_state, memory)

            if step % self.__target_update == 0:
                target_model = copy.deepcopy(self.__model)
                state = self.__env.reset()
                total_reward = 0
                while not done:
                    action = self.select_action(state, 0, target_model)
                    state, reward, done, _ = self.__env.step(action)
                    total_reward += reward
                done = False
                state = self.__env.reset()
                rewards_by_target_updates.append(total_reward)
        return rewards_by_target_updates

    def save(self):
        torch.save(self.__model.state_dict(), 'trained_model_weights.pth')


if __name__ == "__main__":
    site_packages_path = os.getenv('SITE_PACKAGES_PATH')
    if site_packages_path is None:
        raise EnvironmentError("SITE_PACKAGES_PATH environment variable is not set")
    sys.path.append(site_packages_path)
    agent = MountainCarAgent()
    rewards = agent.train()
    agent.save()
