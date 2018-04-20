import numpy as np
import numpy.random as rn

def updateQ(Q, s_t, s_tn, a, a_n, R, alpha, gamma):
    """
    It applies Q-Learning update rule.
    Parameters:
    Q -> Q matrix
    s_tn -> new state
    R -> reward
    a -> action
    """
    Q[s_t, a] = (1 - alpha) * Q[s_t, a] + alpha * (R + gamma*Q[s_tn, a_n])
    return Q

def choose_policy(state):
    """
    It chooses the best action given the current state.
    Paramteres:
    state -> array of possible actions in the current state.
    """
    v_max = np.amax(state)
    indexes = np.arange(len(state))[state == v_max]
    rn.shuffle(indexes)
    return indexes[0]

def gen_policy(Q):
    return [choose_policy(state) for state in Q]