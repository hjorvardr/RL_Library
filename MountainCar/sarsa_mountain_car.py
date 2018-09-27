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


def experiment(n_episodes, max_action, default_policy=False, policy=None, render=False):
    res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = max_action
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
        
        for t in range(max_action):
            if (render):
                env.render()
            
            next_action = agent.act((state[0], state[1]), i_episode)
            new_state, reward, end, _ = env.step(next_action)
            new_state = obs_to_state(env, new_state, n_states)
            agent.update_q((state[0], state[1]), (new_state[0], new_state[1]), next_action, reward)
            
            if end:
                if t == max_action - 1:
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
res = experiment(30000, 200)
learnt_policy = np.argmax(res["Q"], axis=2)

#np.savetxt("scores/ql_mountain_car.csv", res["scores"], delimiter=',')

# Testing
res2 = experiment(250, 250, default_policy=True, policy=learnt_policy)
print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(2, 200, default_policy=True, policy=learnt_policy, render=True)
