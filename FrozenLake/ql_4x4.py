import time
import gym
import numpy as np
from tqdm import tqdm
from qlearning_lib import QLAgent

seed = 91

def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.

    Args:
        results: List of 2 elements representing the number of victories and defeats

    Returns:
        results accuracy
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render=False):
    """
    Run a RL experiment that can be either training or testing

    Args:
        n_episodes: number of train/test episodes
        default_policy: boolean to enable testing/training phase
        policy: numpy tensor with a trained policy
        render: enable OpenAI environment graphical rendering

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

    env = gym.make('FrozenLake-v0')
    env.seed(seed)
    
    if (default_policy):
        agent = QLAgent([env.observation_space.n, env.action_space.n], policy=policy)
    else:
        agent = QLAgent([env.observation_space.n, env.action_space.n],
                       epsilon_decay_function=lambda e: e - 0.000036)

    for _ in tqdm(range(n_episodes)):
        state = env.reset()
        cumulative_reward = 0
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()
                time.sleep(1)

            next_action = agent.act(state)
            new_state, reward, end, _ = env.step(next_action)
            if policy is None:
                agent.update_q(state, new_state, next_action, reward)

            if end:
                res[int(reward)] += 1
                steps.append(t)
                cumulative_reward += reward
                scores.append(cumulative_reward)
                break
            else:
                state = new_state
                cumulative_reward += reward

    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "Q": agent.Q}


# Training
train_res = experiment(20000)
learnt_policy = np.argmax(train_res["Q"], axis=1)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()
np.save('ql_4x4_policy.npy', learnt_policy)
#print("Policy learnt: ", learnt_policy)

# np.savetxt("results/training/ql_4x4.csv", train_res["scores"], delimiter=',')

# Testing
test_agent = np.load('ql_4x4_policy.npy')
test_res = experiment(500, default_policy=True, policy=test_agent)
testing_accuracy = accuracy(test_res["results"])
testing_mean_steps = test_res["steps"].mean()
testing_mean_score = test_res["scores"].mean()

# np.savetxt("results/testing/ql_4x4.csv", test_res["scores"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
#experiment(5, default_policy=True, policy=learnt_policy, render=True)