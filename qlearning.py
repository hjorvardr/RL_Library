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
