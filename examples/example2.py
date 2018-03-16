# For practice purpose as part of Reinforcement Learning course.

## Optimized policy via Genetic Algorithm

import gym
import time
import os

acc = 0;

new_policy = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
#new_policy = [3, 1, 0, 2, 1, 3, 1, 2, 2, 2, 1, 1, 0, 0, 1,1] 
best_policy = [0, 3, 3, 3, 0, 3, 0, 1, 3, 1, 0, 0, 2, 2, 1, 1] 
from gym import wrappers
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1', force=True)
for i_episode in range(100000000):
    observation = env.reset()
    for t in range(100):
        env.render()
        observation, reward, done, info = env.step(best_policy[observation])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            input(acc)
            break	
time.sleep(1)
env.close()
