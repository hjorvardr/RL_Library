import os
import time
import gym
import numpy as np
import keras.optimizers 
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from tqdm import tqdm
from random import randint
from dqn_lib import DQNAgent
from enum import Enum
from ensembler import *

def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render = False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    #env = env.unwrapped
    #env.seed(91)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
        
    agent1 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model01", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent2 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model02", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent3 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model35", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent4 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model33", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent5 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model33", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent6 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi1", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent7 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi2", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent8 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="modelprob1", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent9 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="modelprob2", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    
    agents = [agent6, agent3, agent2, agent2, agent1, agent7]
    agents = [agent9]

    agent = EnsemblerAgent(output_dim, agents, EnsemblerType.MAJOR_VOTING_BASED)
    #agent = EnsemblerAgent(output_dim, agents, EnsemblerType.AGGREGATION_BASED)
    #agent = EnsemblerAgent(output_dim, agents, EnsemblerType.TRUST_BASED)
    
    # agent = EnsemblerAgent(output_dim, agents, 0)
    
    for i_episode in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 2])
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()

            next_action = agent.act(state)                       
            new_state, reward, end, _ = env.step(next_action)

            reward = abs(new_state[0] - (-0.5)) # r in [0, 1]
            new_state = np.reshape(new_state, [1, 2])

            
            if end:
                if t == env._max_episode_steps - 1:
                    agent.trust_update(end, 0)
                    res[0] += 1
                else:
                    agent.trust_update(end, 1)
                    res[1] += 1
                    # print("ENTRATO!,", t, "steps")

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward
            


        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent }
    
# Training
res = experiment(100)
print("Accuracy:", accuracy(res["results"]), "Mean steps:", np.mean(res["steps"]), "Mean score:", np.mean(res["scores"]))
# res["agent"].save_model("ensembler01")

# Testing
#res2 = experiment(500, default_policy=True, policy="ensembler01")
#print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(10, render=True, default_policy=True, policy="model1")

