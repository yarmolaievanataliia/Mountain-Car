#!/usr/bin/env python
# coding: utf-8



import sys
sys.path.append('D:\conda\Lib\site-packages')
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
env = gym.make("MountainCar-v0")

device = torch.device("cuda")
def create_new_model():
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    target_model = copy.deepcopy(model)
    model.to(device)
    target_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    return model, target_model, optimizer


    





# In[32]:


gamma = 0.99
def fit(batch, model, target_model, optimizer):
    modified_reward = reward + 10 * abs(new_state[1])
    state, action, reward, next_state, done = batch
    state = torch.tensor(state).to(device).float()
    next_state = torch.tensor(next_state).to(device).float()
    reward = torch.tensor(reward).to(device).float()
    action = torch.tensor(action).to(device)
    done = torch.tensor(done).to(device)
    memory.push((state, action, modified_reward, new_state, done))
    target_q = torch.zeros(reward.size()[0]).float().to(device)
    with torch.no_grad():
        target_q = target_model(next_state).max(1)[0].view(-1) 
        target_q[done] = 0
    target_q = reward + target_q * gamma

    q = model(state).gather(1, action.unsqueeze(1))
    loss = F.mse_loss(q, target_q.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[33]:


def select_action(state, epsilon, model):
    if random.random() < epsilon:
        return random.randint(0, 2)
    return model(torch.tensor(state).to(device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()


# In[34]:


class Memory:
    def init(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return list(zip(*random.sample(self.memory, batch_size)))
    
    def len(self):
        return len(self.memory)


# In[35]:





# In[ ]:




