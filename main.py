import numpy as np
import gym
import random
import torch
from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque

env = UnityEnvironment(file_name='Banana.app')
#env.seed(0)
print('Loaded env')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:\n', state)
state_size = len(state)
print('States have length:', state_size)
seed=0

#env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0

agent = Agent(state_size=state_size, action_size=action_size, seed=seed)

# watch an untrained agent
#state = env.reset()
for j in range(20):
    action = agent.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()
