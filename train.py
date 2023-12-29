#!/usr/bin/env python
# coding: utf-8

# In[31]:


target_update = 1000
batch_size = 128
max_steps = 100001
max_epsilon = 0.5
min_epsilon = 0.1

def train():
      memory = Memory(5000)
      model, target_model, optimizer = create_new_model()
      for step in range(max_steps):
            epsilon = max_epsilon - (max_epsilon - min_epsilon)* step / max_steps
            action = select_action(state, epsilon, model)
            new_state, reward, done, _ = env.step(action)
            memory.push((state, action, reward, new_state, done))
            if done:
                  state = env.reset()
                  done = False
            else:
                  state = new_state
            if step > batch_size:
                 fit(memory.sample(batch_size), model, target_model, optimizer)

            if step % target_update == 0:
                  target_model = copy.deepcopy(model)
                  state = env.reset()
                  total_reward = 0
                  while not done:
                        action = select_action(state, 0, target_model)
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                  done = False
                  state = env.reset()
                  rewards_by_target_updates.append(total_reward)
      return rewards_by_target_updates
    





# In[32]:





# In[33]:





# In[34]:





# In[35]:





# In[ ]:




