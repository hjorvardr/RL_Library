import time
import gym
import numpy as np
from tqdm import tqdm
from sarsa_lib import SARSAAgent


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
        agent = SARSAAgent([n_states, n_states, env.action_space.n], policy=policy)
    else:
        agent = SARSAAgent([n_states, n_states, env.action_space.n])

    for i_episode in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        state = obs_to_state(env, state, n_states)
        cumulative_reward = 0
        agent.extract_policy()
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()
            
            next_action = agent.act((state[0], state[1]), i_episode)
            new_state, reward, end, _ = env.step(next_action)
            new_state = obs_to_state(env, new_state, n_states)
            agent.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward)
            
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
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "Q": agent.Q}


# Training
train_res = experiment(30000)
learnt_policy = np.argmax(train_res["Q"], axis=2)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()
np.save('ql_policy.npy', learnt_policy)

#np.savetxt("scores/ql_mountain_car.csv", res["scores"], delimiter=',')

# Testing
test_agent = np.load('ql_policy.npy')
test_res = experiment(500, default_policy=True, policy=test_agent)
testing_accuracy = accuracy(test_res["results"])
testing_mean_steps = test_res["steps"].mean()
testing_mean_score = test_res["scores"].mean()

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
#experiment(2, 200, default_policy=True, policy=learnt_policy, render=True)