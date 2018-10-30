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


def experiment(n_episodes, testing=False, render = False, agent_config=None):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode

    env = gym.make('MountainCar-v0')
    #env = env.unwrapped
    #env.seed(91)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    if agent_config is None:
        int1 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="17-model23", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
        int2 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="41-model23", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
        int3 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="73-model23", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
        bad1 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi1", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
        # bad2 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi2", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)

        agents = [int1, int2, bad1]
        # agent = EnsemblerAgent(output_dim, agents, EnsemblerType.MAJOR_VOTING_BASED)
        agent = EnsemblerAgent(output_dim, agents, EnsemblerType.TRUST_BASED)
    else:
        agent = agent_config

    for _ in tqdm(range(n_episodes), desc="Episode"):
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
                    if not testing:
                        agent.trust_update(False)
                    res[0] += 1
                else:
                    if not testing:
                        agent.trust_update(True)
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
training_res = experiment(500, testing=False)
print("Training accuracy:", accuracy(training_res["results"]), "Training Mean steps:", \
np.mean(training_res["steps"]), "Training Mean score:", np.mean(training_res["scores"]))

# np.savetxt("results/major.csv", training_res["steps"], delimiter=',')

# Testing
test_res = experiment(500, testing=True, agent_config=training_res["agent"])
print("Testing accuracy: %s, Testing mean score: %s" % (accuracy(test_res["results"]), np.mean(test_res["scores"])))

np.savetxt("results/trust.csv", test_res["steps"], delimiter=',')
