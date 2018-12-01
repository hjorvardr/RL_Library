import os
import time
import gym
from keras import backend as K
from keras.layers import Dense
import keras.optimizers 
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dqn_lib import DQNAgent
from ensembler import *

seed = 91
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
tf.set_random_seed(seed)


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
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    layer1 = Dense(15, input_dim=input_dim, activation='relu')
    layer2 = Dense(output_dim)

    agent1 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent2 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent3 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent4 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent5 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent6 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent7 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent8 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent9 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agent10 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.01, optimizer=keras.optimizers.RMSprop(0.001), tb_dir=None)
    agents = [agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9, agent10]
    agentE = EnsemblerAgent(env.action_space.n, agents, EnsemblerType.MAJOR_VOTING_BASED)

    evaluate = False
    
    for i_episode in tqdm(range(n_episodes + 1), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 2])
        
        if i_episode > 0 and i_episode % 120 == 0:
            evaluate = True
            
        if evaluate == False:
            for t in range(env._max_episode_steps):
                if (render):
                    env.render()

                next_action = agentE.act(state)
                new_state, reward, end, _ = env.step(next_action)
                original_state = new_state

                # r1 = reward + 0.1 * original_state[0]
                # r2 = reward + 0.2 * np.sin(3 * original_state[0])
                # r3 = reward + 0.7 * (original_state[1] * original_state[1])

                r1 = reward + original_state[0]
                r2 = reward + np.sin(3 * original_state[0])
                r3 = reward + (original_state[1] * original_state[1])
                r4 = abs(new_state[0] - (-0.5)) # r in [0, 1]

                new_state = np.reshape(new_state, [1, 2])

                agent1.memoise((state, next_action, r1, new_state, end))
                agent2.memoise((state, next_action, r2, new_state, end))
                agent3.memoise((state, next_action, r1, new_state, end))
                agent4.memoise((state, next_action, r2, new_state, end))
                agent5.memoise((state, next_action, r1, new_state, end))
                agent6.memoise((state, next_action, r2, new_state, end))
                agent7.memoise((state, next_action, r1, new_state, end))
                agent8.memoise((state, next_action, r2, new_state, end))
                agent9.memoise((state, next_action, r1, new_state, end))
                agent10.memoise((state, next_action, r2, new_state, end))

                if end:
                    if t == env._max_episode_steps - 1:
                        res[0] += 1
                    else:
                        res[1] += 1
                        print("ENTRATO!,", t, "steps","reward: ", cumulative_reward)

                    steps.append(t)
                    break
                else:
                    state = new_state
                    cumulative_reward += reward
                
                for agent in agentE.agents:
                    agent.learn()

            cumulative_reward += reward
            scores.append(cumulative_reward)
        else:
            evaluate = False
            eval_res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
            eval_scores = [] # Cumulative rewards
            eval_steps = [] # Steps per episode

            for i_episode in range(100):
                state = env.reset()

                state = np.reshape(state, [1, 2])
                cumulative_reward = 0

                for t in range(env._max_episode_steps):
                    if (render):
                        env.render()

                    next_action = agentE.act(state)
                    new_state, reward, end, _ = env.step(next_action)

                    new_state = np.reshape(new_state, [1, 2])

                    if end:
                        if t == env._max_episode_steps - 1:
                            eval_res[0] += 1
                        else:
                            eval_res[1] += 1

                        eval_steps.append(t)
                        break
                    else:
                        state = new_state
                        cumulative_reward += reward

                cumulative_reward += reward
                eval_scores.append(cumulative_reward)

            testing_accuracy = accuracy(np.array(eval_res))
            testing_mean_steps = np.array(eval_steps).mean()
            testing_mean_score = np.array(eval_scores).mean()
            print("\nTraining episodes:", len(steps), "Training mean score:", np.array(steps).mean(),             "Training mean steps", np.array(scores).mean(), "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores)}


# Training
train_res = experiment(205)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()

# np.savetxt("results/ens_agent8_major.csv", train_res["steps"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps)

# Rendering
#experiment(2, 200, default_policy=True, policy=learnt_policy, render=True)
