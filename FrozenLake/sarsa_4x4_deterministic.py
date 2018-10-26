import time
import numpy as np
import gym
from tqdm import tqdm
from sarsa_lib import SARSAAgent


def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, max_action, default_policy=False, policy=None, render=False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode

    env = gym.make('FrozenLakeNotSlippery-v0')
    env.seed(91)
    
    if (default_policy):
        agent = SARSAAgent([env.observation_space.n, env.action_space.n], policy=policy, alpha=1)
    else:
        agent = SARSAAgent([env.observation_space.n, env.action_space.n], alpha=1)

    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        cumulative_reward = 0
        agent.extract_policy()
        
        for t in range(max_action):
            if (render):
                env.render()
                time.sleep(1)

            next_action = agent.act(state, i_episode)
            new_state, reward, end, _ = env.step(next_action)
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


gym.envs.registration.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

# Training
res = experiment(10000, 100)
learnt_policy = np.argmax(res["Q"], axis=1)
print("Policy learnt: ", learnt_policy)

# np.savetxt("results/sarsa_4x4_deterministic.csv", res["scores"], delimiter=',')

# Testing
res2 = experiment(5000, 1000, default_policy=True, policy=learnt_policy)
print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(5, 1000, default_policy=True, policy=learnt_policy, render=True)