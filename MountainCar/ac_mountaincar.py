import os
import time
import gym
import numpy as np
import keras.optimizers 
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from tqdm import tqdm
from ac_lib import ACAgent


def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render=False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    env.seed(91)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    layer1 = Dense(15, input_dim=input_dim, activation='relu')
    layer2 = Dense(output_dim)
    layer3 = Dense(1)
        
    if default_policy:
        agent = ACAgent(output_dim, None, None, default_policy=True, model_filename=policy, epsilon=0, epsilon_lower_bound=0, learn_thresh=0)
    else:
        agent = ACAgent(output_dim, [layer1, layer2], [layer1, layer3], epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01)

    for _ in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 2])
        
        #for t in tqdm(range(env._max_episode_steps), desc="Action", leave=False):
        for t in range(env._max_episode_steps):
            if (render):
                env.render()

            next_action = agent.act(state)                       
            new_state, reward, end, _ = env.step(next_action)

            reward = abs(new_state[0] - (-0.5)) # r in [0, 1]
            new_state = np.reshape(new_state, [1, 2])
            
            if end:
                if t == env._max_episode_steps - 1:
                    res[0] += 1
                else:
                    res[1] += 1
                    print("ENTRATO!,", t, "steps")

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward
            
            agent.learn(state, next_action, new_state, reward, end)

        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent}


# Training
res = experiment(12000)
res["agent"].save_model("model1")

#np.savetxt("scores/dqn_mountain_car.csv", res["scores"], delimiter=',')

# Testing
res2 = experiment(100, default_policy=True, policy="model1")
print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
experiment(10, render=True, default_policy=True, policy="model1")