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


def experiment(n_episodes, default_policy=False, policy=None, render=False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode

    env = gym.make('FrozenLakeNotSlippery8x8-v0')
    env.seed(91)
    
    if (default_policy):
        agent = SARSAAgent([env.observation_space.n, env.action_space.n], policy=policy, alpha=1)
    else:
        agent = SARSAAgent([env.observation_space.n, env.action_space.n], alpha=1,
                          epsilon_decay_function=lambda e: e - 0.000016)

    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        cumulative_reward = 0
        if not default_policy:
            agent.extract_policy()
        
        for t in range(200):
            if (render):
                env.render()
                time.sleep(1)

            next_action = agent.act(state, i_episode)
            new_state, reward, end, _ = env.step(next_action)
            if policy is None:
                agent.update_q(state, new_state, next_action, reward)

            if end or t == 199:
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
    id='FrozenLakeNotSlippery8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
)

# Training
train_res = experiment(3000)
learnt_policy = np.argmax(train_res["Q"], axis=1)
training_mean_steps = train_res["steps"].mean()
training_mean_score = train_res["scores"].mean()
np.save('sarsa_8x8d_policy.npy', learnt_policy)
#print("Policy learnt: ", learnt_policy)

# np.savetxt("results/training/sarsa_8x8_deterministic.csv", train_res["scores"], delimiter=',')

# Testing
test_agent = np.load('sarsa_8x8d_policy.npy')
test_res = experiment(500, default_policy=True, policy=test_agent)
testing_accuracy = accuracy(test_res["results"])
testing_mean_steps = test_res["steps"].mean()
testing_mean_score = test_res["scores"].mean()

# np.savetxt("results/testing/sarsa_8x8_deterministic.csv", test_res["scores"], delimiter=',')

print("Training episodes:", len(train_res["steps"]), "Training mean score:", training_mean_score, \
"Training mean steps", training_mean_steps, "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

# Rendering
#experiment(5, default_policy=True, policy=learnt_policy, render=True)