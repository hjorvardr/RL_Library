import gym
import os
import numpy as np
import tensorflow as tf
from dqn_lib import DQNAgent
from keras import backend as K
from keras.layers import Dense
from time import time
from tqdm import tqdm

def experiment(n_episodes = 5000, max_action = 100000, default_policy = False, policy = np.zeros(64), render = False):
    
    """
    Execute an experiment given a configuration
    Parameters:
    n_episodes -> number of completed/failed plays
    max_action -> maximum number of actions per episode
    """

    with tf.device('/cpu:0'):
        Res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
        Scores = [] # Cumulative rewards
        Steps = [] # Steps per episode
        
        env = gym.make('CartPole-v0')
        env = env.unwrapped

        if default_policy:
            env._max_episode_steps = max_action
        else:
            env._max_episode_steps = 1000000
        
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        layer1 = Dense(15, input_dim = input_dim, activation = 'relu')
        layer2 = Dense(output_dim)
            
        agent = DQNAgent(input_dim, output_dim, [layer1, layer2], use_ddqn=True)

        for _ in tqdm(range(n_episodes), desc="Episode"):
            state = env.reset()
            cumulative_reward = 0

            state = np.reshape(state,[1,4])
            
            t = 0
            #for t in tqdm(range(env._max_episode_steps), desc="Action", leave=False):
            for t in range(env._max_episode_steps):
                if (render):
                    env.render()

                next_action = agent.act(state)                       
                new_state, reward, end, _ = env.step(next_action)

                x, x_dot, theta, thetadot = new_state
                r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
                reward = r1 + r2

                new_state = np.reshape(new_state,[1,4])
                
                agent.memoise((state, next_action, reward, new_state, end))

                if end:
                    if t == env._max_episode_steps - 1:
                        Res[0] += 1
                    else:
                        Res[1] += 1
                        print("ENTRATO!,", t, "steps")

                    Steps.append(t)
                    break
                else:
                    state = new_state
                    cumulative_reward += reward
                
                agent.learn()
                t += 1

            cumulative_reward += reward
            Scores.append(cumulative_reward)
        env.close()
        return {"results": np.array(Res), "steps": np.array(Steps), "scores": np.array(Scores), "agent": agent }
    
config = {"n_episodes": 10000, "max_action": 10000, "render": False}
res = experiment(**config)
res["agent"].save_model("model1")