import os
import time
import gym
import keras.optimizers 
from keras import backend as K
from keras.layers import Dense
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from dqn_lib import DQNAgent

os.environ['PYTHONHASHSEED'] = '0'
seed = 17
np.random.seed(seed)
tf.set_random_seed(seed)

def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.

    Args:
        results: List of 2 elements representing the number of victories and defeats

    Returns:
        results accuracy
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render=False, agent_config=None):
    """
    Run a RL experiment that can be either training or testing

    Args:
        n_episodes: number of train/test episodes
        default_policy: boolean to enable testing/training phase
        policy: numpy tensor with a trained policy
        render: enable OpenAI environment graphical rendering
        agent_config: DQNAgent object

    Returns:
        Dictionary with:
            cumulative experiments outcomes
            list of steps per episode
            list of cumulative rewards
            trained policy
    """
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    if agent_config is None:
        if default_policy:
            agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy,
                            epsilon=0, epsilon_lower_bound=0, learn_thresh=0)
        else:
            layer1 = Dense(15, input_dim=input_dim, activation='relu')
            layer2 = Dense(output_dim)
            agent = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300,
                            epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01,
                            optimizer=keras.optimizers.RMSprop(0.001))
    else:
        agent = agent_config


    for i_episode in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        # Model validation for early stopping
        if i_episode > 0 and (i_episode % 100) == 0 and not default_policy:
            agent.save_model("tmp_model")
            evaluation_result = experiment(500, default_policy=True, policy="tmp_model")
            acc = accuracy(evaluation_result["results"])
            if acc == 100:
                break
            else:
                print("Accuracy:", acc, "Episode:", i_episode)

        state = np.reshape(state, [1, 2])
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()

            next_action = agent.act(state)                       
            new_state, reward, end, _ = env.step(next_action)

            reward = abs(new_state[0] - (-0.5)) # r in [0, 1] (reward shaping)
            new_state = np.reshape(new_state, [1, 2])
            
            agent.memoise((state, next_action, reward, new_state, end))

            if end:
                if t == env._max_episode_steps - 1:
                    res[0] += 1
                else:
                    res[1] += 1
                    # print("ENTRATO!,", t, "steps")

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward
            
            agent.learn()

        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent}


# Training
# res = experiment(120)
# res["agent"].save_model("model1")

# Testing
# res2 = experiment(100, default_policy=True, policy="model1")
# print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(10, render=True, default_policy=True, policy="model1")

input_dim = 2
output_dim = 3

experiments = []

layer1 = Dense(15, input_dim=input_dim, activation='relu')
layer2 = Dense(output_dim)
layers = [layer1, layer2]
experiments.append(("model23", 25000, DQNAgent(output_dim, layers, use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e * 0.995, epsilon_lower_bound=0.01, optimizer=keras.optimizers.Adam(0.001), tb_dir=None)))

def train_and_test(experiments):
    df = pd.DataFrame(columns=['model name', 'episode number', 'train mean score', 'train mean steps', 'test accuracy', 'test mean score', 'test mean steps'])
    for model_name, steps, train_agent in experiments:
        # Train
        train_res = experiment(steps, agent_config=train_agent)
        train_res["agent"].save_model(model_name)
        training_mean_steps = train_res["steps"].mean()
        training_mean_score = train_res["scores"].mean()

        np.savetxt("results/training/ddqn.csv", train_res["steps"], delimiter=',')

        # Test
        test_agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=model_name, epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
        test_res = experiment(500, default_policy=True, policy=model_name, agent_config=test_agent)
        testing_accuracy = accuracy(test_res["results"])
        testing_mean_steps = test_res["steps"].mean()
        testing_mean_score = test_res["scores"].mean()
        
        np.savetxt("results/testing/ddqn.csv", test_res["steps"], delimiter=',')

        df.loc[len(df)] = [model_name, len(train_res["steps"]), training_mean_score, training_mean_steps, testing_accuracy, testing_mean_score, testing_mean_steps]

    df.to_csv('experiments.csv')

def main():
    train_and_test(experiments)

if __name__ == "__main__":
    main()
