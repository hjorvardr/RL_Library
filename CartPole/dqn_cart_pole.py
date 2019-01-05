import os
import time
import gym
import keras.optimizers 
from keras import backend as K
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dqn_lib import DQNAgent

os.environ['PYTHONHASHSEED'] = '0'
seed = 73
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


def experiment(n_episodes, default_policy=False, policy=None, render = False):
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
    steps = [] # steps per episode
    
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
        
    if default_policy:
        agent = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename=policy,
                        epsilon=0, epsilon_lower_bound=0, learn_thresh=0, tb_dir=None)
    else:
        layer1 = Dense(10, input_dim=input_dim, activation='relu')
        layer2 = Dense(output_dim)
        agent = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=2000, update_rate=100,
                        epsilon_decay_function=lambda e: e - 0.001, epsilon_lower_bound=0.1,
                        optimizer=keras.optimizers.RMSprop(0.001), memory_size=2000, tb_dir=None)

    for _ in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 4])
        
        t = 0
        while True:
            if (render):
                env.render()
                time.sleep(0.1)

            next_action = agent.act(state)
            new_state, reward, end, _ = env.step(next_action)

            x, x_dot, theta, theta_dot = new_state
            new_state = np.reshape(new_state, [1, 4])
            
            # Reward shaping
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r3 = -abs(theta_dot)
            reward = r1 + r2 + r3
            
            agent.memoise((state, next_action, reward, new_state, end))

            if end or t > 199:
                if  t < 195:
                    res[0] += 1
                else:
                    res[1] += 1
                    # print("ENTRATO!,", t, "steps","reward: ",cumulative_reward)

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
train_res = experiment(500)
train_res["agent"].save_model("ddqn")
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()

# np.savetxt("results/training/ddqn.csv", train_res["steps"], delimiter=',')

# Testing
test_res = experiment(500, default_policy=True, policy="ddqn")
testing_accuracy = accuracy(test_res["results"])
testing_mean_steps = test_res["steps"].mean()
testing_mean_score = test_res["scores"].mean()

# np.savetxt("results/testing/ddqn.csv", test_res["steps"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
# experiment(1, render=True, default_policy=True, policy="model_cp")
