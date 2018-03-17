import gym
import time
import os

import numpy as np


"""
SFFF
FHFH
FFFH
HFFG
[0, x, x, x, 0, x, 1, x, 2, 1, 0, x, x, 3, 1, x]  <- our gamma .9
[0, 3, 3, 3, 0, 3, 0, 1, 3, 1, 0, 0, 2, 2, 1, 1]  <- GA best
[3, 3, 1, 1, x, x, 1, x, 1, 1, 1, x, x, 2, 1, x]  <- gamma 0.2
[1, 1, 0, x, 2, x, 1, x, 2, 1, 0, x, x, 3, 2, x]  <- gamma 1.0
[1, 1, 0, x, 0, x, 0, x, 3, 2, 1, x, x, 0, 2, x]  <- gamma .5

"""

gamma = 0.2
alpha = 0.5

def updateQ(Q, s_t, s_tn, a, R):
    """
    Q -> Q matrix
    s_tn -> new state
    R -> reward
    a -> action
    """
    a_max = np.argmax(Q[s_tn])
    Q[s_t, a] = (1 - alpha)*Q[s_t, a] + alpha * (R + gamma*Q[s_tn, a_max])
    return Q

def choose_policy(states):
    import numpy.random as rn
    #print(states)
    v_max = np.amax(states)
    #print(v_max)
    indici = np.arange(len(states))[states == v_max]
    #print(indici)
    rn.shuffle(indici)
    return indici[0]


Q = np.ones((16, 4))

print("Q(t:0)")
print(Q)

acc = 0;

#best_policy = [0, 3, 3, 3, 0, 3, 0, 1, 3, 1, 0, 0, 2, 2, 1, 1] 
from gym import wrappers
env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1', force=True)
for i_episode in range(100000):
    s_old = env.reset()

    for t in range(100):
        #env.render()
        policy = choose_policy(Q[s_old])
        s_new, reward, done, info = env.step(policy)
        Q = updateQ(Q, s_old, s_new, policy, reward)
        s_old = s_new
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            
            #input(acc)
            break	
time.sleep(1)
print(Q)
env.close()
