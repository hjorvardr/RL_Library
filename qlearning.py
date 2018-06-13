import numpy as np
import numpy.random as rn

def updateQ(Q, state, new_state, action, reward, alpha, gamma):
    """
    It applies Q-Learning update rule.
    Parameters:
    Q -> Q matrix
    state -> current state t
    new_state -> next state t
    reward -> reward
    action -> current action
    """
    future_action = np.argmax(Q[new_state]) # Find the best action to perform at time t+1
    Q[state, action] = (1 - alpha)*Q[state, action] + alpha * (reward + gamma*Q[new_state, future_action])
    return Q

def updateQ_tensor(Q, state, new_state, action, reward, alpha, gamma):
    """
    It applies Q-Learning update rule considering 3-dimensional matrices. It is used in MountainCar-v0 environment.
    Parameters:
    Q -> Q matrix
    state -> current state t
    new_state -> next state t
    reward -> reward
    action -> current action
    """
    future_action = np.argmax(Q[new_state[0],new_state[1]]) # Find the best action to perform at time t+1
    Q[state[0],state[1], action] = (1 - alpha)*Q[state[0],state[1], action] + alpha * (reward + gamma*Q[new_state[0],new_state[1], future_action])
    return Q

def next_action1(state):
    """
    It chooses the best action given the current state.
    Paramteres:
    state -> array of possible actions in the current state.
    """
    max_value = np.amax(state)
    max_indexes = np.arange(len(state))[state == max_value]
    rn.shuffle(max_indexes)
    return max_indexes[0]

def next_action2(state,i_episode):
    return np.argmax(state + np.random.randn(1,len(state))*(1./(i_episode+1)))

def next_action3(state,epsilon):
    """
    It chooses the best action given the current state.
    Paramteres:
    state -> array of possible actions in the current state.
    """
    if np.random.uniform() > epsilon:
        max_value = np.amax(state)
        max_indexes = np.arange(len(state))[state == max_value]
        rn.shuffle(max_indexes)
        return max_indexes[0]
    return np.argmax(np.random.uniform(0,1, size=4))

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
    