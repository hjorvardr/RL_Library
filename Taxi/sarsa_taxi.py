import time
import gym
import numpy as np
from tqdm import tqdm
from sarsa_lib import SARSAAgent

seed = 91

def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render=False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode

    env = gym.make('Taxi-v2')
    env.seed(seed)
    
    if (default_policy):
        agent = SARSAAgent([env.observation_space.n, env.action_space.n], policy=policy)
    else:
        agent = SARSAAgent([env.observation_space.n, env.action_space.n],
                          epsilon_decay_function=lambda e: e - 0.000016, update_rate=10)

    for _ in tqdm(range(n_episodes)):
        state = env.reset()
        cumulative_reward = 0

        if not default_policy:
            agent.extract_policy()
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()
                time.sleep(1)
            
            next_action = agent.act(state)
            new_state, reward, end, _ = env.step(next_action)
            if policy is None:
                agent.update_q(state, new_state, next_action, reward)

            if reward == -10:
                res[0] += 1
            if reward == 20:
                res[1] += 1
                
            if end:
                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward

        cumulative_reward += reward
        scores.append(cumulative_reward)

    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "Q": agent.Q}


# Training
train_res = experiment(10000)
learnt_policy = np.argmax(train_res["Q"], axis=1)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()
np.save('sarsa_policy.npy', learnt_policy)

# np.savetxt("results/training/sarsa.csv", train_res["scores"], delimiter=',')

# Testing
test_agent = np.load('sarsa_policy.npy')
test_res = experiment(10000, default_policy=True, policy=test_agent)
testing_accuracy = accuracy(test_res["results"])
testing_mean_steps = test_res["steps"].mean()
testing_mean_score = test_res["scores"].mean()

# np.savetxt("results/testing/sarsa.csv", test_res["scores"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
#experiment(5, default_policy=True, policy=learnt_policy, render=True)

