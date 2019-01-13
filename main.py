import numpy as np
import gym
import random
import torch
from tqdm import tqdm
from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque

env = UnityEnvironment(file_name='Banana.app')
# env.seed(0)
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
seed = 0

# env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0

agent = Agent(state_size=state_size, action_size=action_size, seed=seed)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning

    Params
    ======
        n_episodes (int): max number of training episodes
        max_t (int): max number of timesteps per episode
        eps_start (float): start value of epsilon, for epsilon-greedy action selection
        eps_end (float): min value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in tqdm(range(1, n_episodes+1)):
        state = env.reset()
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            results = env.step(action)
            results= results['BananaBrain']
            next_state, reward, done = results.vector_observations, results.rewards[0], results.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > 15:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()
env.close()
