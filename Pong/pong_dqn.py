import random as ran
import os
import time
import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
from dqn_lib import DQNAgent


# Original size: 210x160x3
def pre_processing(observe):
    grayscaled = rgb2gray(observe) # 210x160
    grayscaled = grayscaled[16:201,:]
    processed_observe = np.uint8(resize(grayscaled, (84, 84), mode='constant') * 255)
    return processed_observe


def experiment(n_episodes, default_policy=False, policy=None, render=False):

    with tf.device('/gpu:0'):
        res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
        scores = [] # Cumulative rewards
        steps = [] # Steps per episode
        
        env = gym.make('PongNoFrameskip-v4')

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        layers = [Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
                  Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
                  Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
                  Flatten(),
                  Dense(512, activation='relu'),
                  Dense(output_dim)]
            
        if default_policy:
            agent = DQNAgent(input_dim, output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy, epsilon=0.05, epsilon_lower_bound=0.05)
        else:
            agent = DQNAgent(input_dim, output_dim, layers, use_ddqn=True, memory_size=100000, gamma=0.99, learn_thresh=10000,
                            epsilon_lower_bound=0.02, epsilon_decay_function=lambda e: e - (0.98 / 100000), update_rate=1000,
                            optimizer=Adam(0.0001))

        gathered_frame = 0
        for episode_number in tqdm(range(n_episodes), desc="Episode"):
            frame = env.reset()
            cumulative_reward = 0
            
            state = pre_processing(frame)
            empty_state = np.zeros(state.shape)
            stack = np.stack((empty_state, empty_state, empty_state, empty_state), axis=2)
            stack = np.reshape([stack], (1, 84, 84, 4))

            next_action = [1, 4, 5][ran.randint(0, 2)]
            frame,reward,end,_ = env.step(next_action)
            if (render):
                env.render()
                # time.sleep(0.05)

            new_state = np.reshape(pre_processing(frame), (1, 84, 84, 1))
            new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
            agent.memoise((stack, next_action, reward, new_stack, end))

            t = 0
            for _ in range(ran.randint(1, 30)):
                next_action = 0
                gathered_frame += 1
                frame, reward, end, _ = env.step(next_action)
                new_state = np.reshape(pre_processing(frame), (1, 84, 84, 1))
                new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
                agent.memoise((stack, next_action, reward, new_stack, end))
                stack = new_stack
                if (render):
                    env.render()
                    # time.sleep(0.05)
            
            while True:
                next_action = agent.act(stack)
                new_state, reward, end, _ = env.step(next_action)
                if (render):
                    env.render()
                    # time.sleep(0.05)
                reward = np.clip(reward, -1., 1.)

                cumulative_reward += reward

                new_state = np.reshape(pre_processing(new_state), (1, 84, 84, 1))
                new_stack = np.append(new_state, stack[:, :, :, :3], axis=3)
                agent.memoise((stack, next_action, reward, new_stack, end))

                stack = new_stack
                gathered_frame += 1

                if end:
                    if cumulative_reward > 0:
                        res[1] += 1
                        print("You Won!, steps:", t, "reward:", cumulative_reward, "frames:", gathered_frame)
                    else:
                        res[0] += 1
                        print("You Lost!, steps:", t, "reward:", cumulative_reward, "frames:", gathered_frame)
                    steps.append(t)
                    break
                
                agent.learn()
                t += 1

            scores.append(cumulative_reward)
            if episode_number > 0 and episode_number % 10 == 0:
                agent.save_model("partial_model_pong")

        
        env.close()
        return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent}
    
# Training
res = experiment(500, render=False)
res["agent"].save_model("model500eps")

# Testing
res = experiment(20, render=True, default_policy=True, policy="SavedNetworks/partial_model_pong")
