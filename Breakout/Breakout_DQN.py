import os
import random as ran
import time
import gym
from keras import backend as K
from keras.initializers import VarianceScaling
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf
from tqdm import tqdm
from dqn_lib import DQNAgent


# Original size: 210x160x3
def pre_processing(observe):
    grayscaled = rgb2gray(observe) # 210x160
    grayscaled = grayscaled[16:201,:]
    processed_observe = np.uint8(resize(grayscaled, (84, 84), mode='constant') * 255)
    return processed_observe


# 0: stay, 1: start, 2: right, 3: left
def experiment(n_episodes, max_action, default_policy=False, policy=None, render=False):

    with tf.device('/gpu:0'):
        res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
        scores = [] # Cumulative rewards
        steps = [] # Steps per episode
        
        env = gym.make('BreakoutDeterministic-v4')

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
            
        if default_policy:
            agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy, epsilon=0.05, epsilon_lower_bound=0.05)
        else:
            layers = [Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4), kernel_initializer=VarianceScaling(scale=2.0)),
                    Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=VarianceScaling(scale=2.0)),
                    Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=VarianceScaling(scale=2.0)),
                    Flatten(),
                    Dense(512, activation='relu', kernel_initializer=VarianceScaling(scale=2.0)),
                    Dense(output_dim)]
            agent = DQNAgent(output_dim, layers, use_ddqn=True, memory_size=720000, gamma=0.99)

        gathered_frame = 0
        for episode_number in tqdm(range(n_episodes), desc="Episode"):
            frame = env.reset()
            state = pre_processing(frame)
            empty_state = np.zeros(state.shape, dtype="uint8")
            cumulative_reward = 0
                
            has_lost_life = True
            start_life = env.unwrapped.ale.lives()

            t = 0
            while True:
                if has_lost_life:
                    next_action = 1

                    stack = np.stack((empty_state, empty_state, empty_state, empty_state), axis=2)
                    stack = np.reshape([stack], (1, 84, 84, 4))

                    for _ in range(ran.randint(1, 10)):
                        gathered_frame += 1
                        frame, reward,end,_ = env.step(next_action)
                        new_state = np.reshape(pre_processing(frame), (1, 84, 84, 1))
                        new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
                        agent.memoise((stack, next_action, reward, new_state, end))
                        stack = new_stack

                        if (render):
                            env.render()

                    has_lost_life = False

                next_action = agent.act(stack)
                new_state, reward, end, info = env.step(next_action)

                if (render):
                    env.render()

                reward = np.clip(reward, -1., 1.)

                if info['ale.lives'] < start_life:
                    has_lost_life = True
                    start_life = info['ale.lives']
                    res[0] += 1

                cumulative_reward += reward

                new_state = np.reshape(pre_processing(new_state), (1, 84, 84, 1))
                new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
                agent.memoise((stack, next_action, reward, new_state, has_lost_life))

                stack = new_stack
                gathered_frame += 1

                if end:
                    if not has_lost_life:
                        res[1] += 1
                        print("You Won!, steps:", t, "reward:", cumulative_reward, "frames:", gathered_frame)
                    else:
                        print("You Lost!, steps:", t, "reward:", cumulative_reward, "frames:", gathered_frame)
                    steps.append(t)
                    break
                
                agent.learn()
                t += 1

            scores.append(cumulative_reward)
            if episode_number >= 100 and episode_number % 50 == 0:
                model_name = "partial_model_breakout" + str(episode_number)
                agent.save_model(model_name)
        
        env.close()
        return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent}
    
# Training
res = experiment(100000, 10000000, render=False)
res["agent"].save_model("ddqn")

# Testing
res = experiment(20, 10000000, render=True, default_policy=True, policy="ddqn")
