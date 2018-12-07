from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='Banana.app')
print('got env')

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

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0
