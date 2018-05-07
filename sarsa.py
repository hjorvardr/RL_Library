import numpy as np
import numpy.random as rn

def updateQ(Q, state, new_state, action, next_action, reward, alpha, gamma):
    """
    It applies Q-Learning update rule.
    Parameters:
    Q -> Q matrix
    state -> current state t
    new_state -> next state t
    reward -> reward
    action -> current action
    next_action -> next action
    """
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma*Q[new_state, next_action])
    return Q

def next_action1(state):
    """
    It chooses the best action given the current state.
    Paramteres:
    state -> array of possible actions in the current state.
    """
    v_max = np.amax(state)
    indexes = np.arange(len(state))[state == v_max]
    rn.shuffle(indexes)
    return indexes[0]

def next_action2(state,i_episode):
    return np.argmax(state + np.random.randn(1,len(state))*(1./(i_episode+1)))

def next_action3(action,epsilon):
    """
    It chooses the best action given the current state.
    Paramteres:
    action -> best action to perform.
    epsilon -> exploration/exploitation probability.
    """
    if np.random.uniform() > epsilon:
        return action
    return np.argmax(np.random.uniform(0,1, size=6))

def gen_policy(Q):
    return [next_action1(state) for state in Q]

def get_epsilon(k,n):
    res = (n - k) / n
    if res < 0.01:
        return 0.01
    return res

def get_epsilon_exp(n):
    res = 1 / (n + 1)
    if res < 0.01:
        return 0.01
    return res