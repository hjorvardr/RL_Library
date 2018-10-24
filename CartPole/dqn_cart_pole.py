import os
import time
import gym
import numpy as np
import keras.optimizers 
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from tqdm import tqdm
from dqn_lib import DQNAgent

os.environ['PYTHONHASHSEED'] = '0'

seed = 73
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(seed)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

# random.seed(seed)

tf.set_random_seed(seed)


def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render = False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # steps per episode
    
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env._max_episode_steps = 10000000000000
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    layer1 = Dense(10, input_dim=input_dim, activation='relu')
    layer2 = Dense(output_dim)
        
    if default_policy:
        agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy, epsilon=0, epsilon_lower_bound=0, learn_thresh=0)
    else:
        agent = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=2000, update_rate=100, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.1, optimizer=keras.optimizers.RMSprop(0.001), memory_size=2000)

    for _ in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 4])
        
        t = 0
        while True:
        #for t in range(max_action):

            if (render):
                env.render()
                time.sleep(0.1)

            next_action = agent.act(state)
                    
            new_state, reward, end, _ = env.step(next_action)

            #print(reward)
            x, x_dot, theta, theta_dot = new_state
            new_state = np.reshape(new_state,[1,4])
            
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            #if t > 300:
            #    end = True
            agent.memoise((state, next_action, reward, new_state, end))

            if end:# or t > 300:
                if  t < 195:
                    res[0] += 1
                    #reward = reward - 100
                    #memory.append((state, next_action, reward, new_state, end))
                else:
                    res[1] += 1
                    print("ENTRATO!,", t, "steps","reward: ",cumulative_reward)

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward

            agent.learn()
            t += 1

        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent }
    
# Training
res = experiment(10000)
res["agent"].save_model("model1")

# Testing
res2 = experiment(100, default_policy=True, policy="model1")
print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(10, render=True, default_policy=True, policy="model1")