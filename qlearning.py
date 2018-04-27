import numpy as np
import numpy.random as rn

def updateQ(Q, s_t, s_tn, a, R, alpha, gamma):
    """
    It applies Q-Learning update rule.
    Parameters:
    Q -> Q matrix
    s_tn -> new state
    R -> reward
    a -> action
    """
    a_max = np.argmax(Q[s_tn])
    Q[s_t, a] = (1 - alpha)*Q[s_t, a] + alpha * (R + gamma*Q[s_tn, a_max])
    #Q[s_t,a] = Q[s_t,a] + alpha*(R + gamma*np.max(Q[s_tn,:]) - Q[s_t,a])
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

def choose_policy_greedy(state,env,i_episode):
    return np.argmax(state + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))