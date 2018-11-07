import time
import gym
import numpy as np
from tqdm import tqdm
import os
import random as ran
import numpy as np
import tensorflow as tf
from qlearning_lib import QLAgent
from ensembler import *

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(91)
tf.set_random_seed(91)

def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def obs_to_state(env, obs, n_states):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


def experiment(n_episodes, default_policy=False, policy=None, render=False):
    res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    env.seed(91)
    n_states = 150
    if (default_policy):
        agentE = QLAgent([n_states, n_states, env.action_space.n], policy=policy, epsilon=0.01, epsilon_lower_bound=0.01)
    else:
        agent0 = QLAgent([n_states, n_states, env.action_space.n], epsilon_decay_function=lambda e: e * 0.995, epsilon_lower_bound=0.01)
        agent1 = QLAgent([n_states, n_states, env.action_space.n], epsilon_decay_function=lambda e: e * 0.995, epsilon_lower_bound=0.01)
        agent2 = QLAgent([n_states, n_states, env.action_space.n], epsilon_decay_function=lambda e: e * 0.995, epsilon_lower_bound=0.01)
        agent3 = QLAgent([n_states, n_states, env.action_space.n], epsilon_decay_function=lambda e: e * 0.995, epsilon_lower_bound=0.01)
        agents = [agent1, agent2, agent3]
        agentE = EnsemblerAgent(env.action_space.n, agents, EnsemblerType.TRUST_BASED)

    evaluate = False
    
    for i_episode in tqdm(range(n_episodes + 1), desc="Episode"):
        state_original = env.reset()
        
        state = obs_to_state(env, state_original, n_states)
        cumulative_reward = 0
        
        if i_episode > 0 and i_episode % 1000 == 0:
            evaluate = True
            
        if evaluate == False:
            for t in range(env._max_episode_steps):
                if (render):
                    env.render()

                next_action = agentE.act((state[0], state[1]))
                state_original, reward, end, _ = env.step(next_action)
                new_state = obs_to_state(env, state_original, n_states)



                if default_policy is False:
                    #agent0.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward)
                    agent1.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward + 0.1 * state_original[0])
                    agent2.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward + 0.2 * np.sin(3 * state_original[0]))
                    agent3.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward + 0.7 * (state_original[1] * state_original[1]))

                if end:
                    if t == env._max_episode_steps - 1:
                        res[0] += 1
                    else:
                        res[1] += 1


                    steps.append(t)
                    break
                else:
                    state = new_state
                    cumulative_reward += reward

                cumulative_reward += reward
                scores.append(cumulative_reward)
            env.close()
        else:
            evaluate = False
            eval_res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
            eval_scores = [] # Cumulative rewards
            eval_steps = [] # Steps per episode

            for i_episode in range(500):
                state_original = env.reset()

                state = obs_to_state(env, state_original, n_states)
                cumulative_reward = 0

                for t in range(env._max_episode_steps):
                    if (render):
                        env.render()

                    next_action = agentE.act((state[0], state[1]))
                    state_original, reward, end, _ = env.step(next_action)
                    new_state = obs_to_state(env, state_original, n_states)

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
            env.close()

            testing_accuracy = accuracy(np.array(eval_res))
            testing_mean_steps = np.array(eval_steps).mean()
            testing_mean_score = np.array(eval_scores).mean()
            print("\nTraining episodes:", len(steps), "Training mean score:", np.array(steps).mean(),             "Training mean steps", np.array(scores).mean(), "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

    return 0 # {"results": np.array(res), "steps": np.array(eval_steps), "scores": np.array(eval_scores), "Q": agent0.Q}


# Training
train_res = experiment(30000)
#learnt_policy = np.argmax(train_res["Q"], axis=2)
#training_mean_steps = train_res["steps"].mean()
#training_mean_score = train_res["scores"].mean()
#np.save('ql_policy.npy', learnt_policy)

# np.savetxt("results/ql.csv", train_res["steps"], delimiter=',')

# Testing
#test_agent = np.load('ql_policy.npy')
#test_res = experiment(500, default_policy=True, policy=test_agent)
#testing_accuracy = accuracy(test_res["results"])
#testing_mean_steps = test_res["steps"].mean()
#testing_mean_score = test_res["scores"].mean()

# np.savetxt("results/ql_test.csv", test_res["steps"], delimiter=',')

#print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
#"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
#experiment(2, 200, default_policy=True, policy=learnt_policy, render=True)
