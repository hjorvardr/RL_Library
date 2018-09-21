import random as ran
import os
import time
import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
from dqn_lib import DQNAgent


# Original size: 210x160x3
def pre_processing(observe):
    grayscaled = rgb2gray(observe) # 210x160
    processed_observe = np.uint8(resize(grayscaled, (84, 84), mode='constant') * 255)
    return processed_observe


# 0: stay
# 1: start
# 2: right
# 3: left

def experiment(n_episodes, max_action, default_policy=False, policy=None, render=False):

    with tf.device('/cpu:0'):
        res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
        scores = [] # Cumulative rewards
        steps = [] # Steps per episode
        
        env = gym.make('BreakoutDeterministic-v4')

        if default_policy:
            env._max_episode_steps = 500000
        else:
            env._max_episode_steps = 1000000
        
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        layers = [Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
                  Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
                  Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
                  Flatten(),
                  Dense(512, activation='relu'),
                  Dense(output_dim)]
            
        if default_policy:
            agent = DQNAgent(input_dim, output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy)
        else:
            agent = DQNAgent(input_dim, output_dim, layers, use_ddqn=True)

        for _ in tqdm(range(n_episodes), desc="Episode"):
            frame = env.reset()
            cumulative_reward = 0

            state = pre_processing(frame)
            stack = np.stack((state, state, state, state), axis=2)
            stack = np.reshape([stack], (1, 84, 84, 4))

            for _ in range(ran.randint(1, 4)):
               _,_,_,_ = env.step(1)
            
            start_life = 5
            dead = False
            t = 0
            #for t in tqdm(range(env._max_episode_steps), desc="Action", leave=False):
            for t in range(env._max_episode_steps):
                if (render):
                    env.render()
                    # time.sleep(2)

                next_action = agent.act(stack)
                new_state, reward, end, info = env.step(next_action)
                # reward = np.clip(reward, -1., 1.)

                # print(next_action, reward)

                new_state = np.reshape(pre_processing(new_state), (1, 84, 84, 1))
                new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
                #new_stack = np.insert(stack, 0, new_state, axis=3)[:, :, :, :4]
                
                agent.memoise((stack, next_action, reward, new_stack, end))

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                    res[0] += 1

                if end:
                    if not dead:
                        res[1] += 1
                        print("ENTRATO!,", t, "steps")

                    steps.append(t)
                    break
                else:
                    # if agent is dead, then reset the history
                    if dead and start_life == 0:
                        break
                    elif dead:
                        dead = False
                    else:
                        stack = new_stack
                
                agent.learn()
                #t += 1

            cumulative_reward += reward
            scores.append(cumulative_reward)
        env.close()
        return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent}
    
# Training
# res = experiment(10, 10000000, render=True)
# res["agent"].save_model("model1")

# Testing
res = experiment(10, 10000000, render=True, default_policy=True, policy="SavedNetworks/model50ep")
