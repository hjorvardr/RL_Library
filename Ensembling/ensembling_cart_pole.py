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
from ensembler import *

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
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    layer1 = Dense(10, input_dim=input_dim, activation='relu')
    layer2 = Dense(output_dim)
        
    if default_policy:
        agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy, epsilon=0, epsilon_lower_bound=0, learn_thresh=0)
    else:
        agent1 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=2000, update_rate=100, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.1, optimizer=keras.optimizers.RMSprop(0.001), memory_size=2000, tb_dir=None)
        agent2 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=2000, update_rate=100, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.1, optimizer=keras.optimizers.RMSprop(0.001), memory_size=2000, tb_dir=None)
        agentE = EnsemblerAgent(output_dim, [agent1, agent2], EnsemblerType.RANK_VOTING_BASED)

    for _ in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 4])
        
        t = 0
        while True:
            if (render):
                env.render()
                time.sleep(0.1)

            next_action = agentE.act(state)
                    
            new_state, reward, end, _ = env.step(next_action)

            x, x_dot, theta, theta_dot = new_state
            new_state = np.reshape(new_state, [1, 4])
            
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            
            agent1.memoise((state, next_action, r1, new_state, end))
            agent2.memoise((state, next_action, r2, new_state, end))

            if end or t > 199:
                if  t < 195:
                    res[0] += 1
                else:
                    res[1] += 1
                    print("ENTRATO!,", t, "steps","reward: ",cumulative_reward)

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward

            agent1.learn()
            agent2.learn()
            t += 1

        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores)}

# Training
train_res = experiment(300)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()

np.savetxt("results/ddqn_ens.csv", train_res["steps"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps)

# Rendering
# experiment(1, render=True, default_policy=True, policy="model1")