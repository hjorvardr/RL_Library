import numpy as np


class QLAgent:

    def __init__(self, q_height, q_width, alpha=0.8, gamma=0.95, policy=None):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.Q = np.zeros([q_height, q_width])
        self.epsilon = 0
        self.epsilon_lower_bound = 0.01
        self.policy = policy
        self.actions = q_width
        np.random.seed(91)

    def update_q(self, state, new_state, action, reward):
        """
        It applies Q-Learning update rule.
        Parameters:
        state -> current state t
        new_state -> next state t
        reward -> reward
        action -> current action
        """
        future_action = np.argmax(self.Q[new_state]) # Find the best action to perform at time t+1
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (reward + self.gamma * self.Q[new_state, future_action])

    def act(self, state, episode_number):
        if (self.policy is not None):
            next_action = self.policy[state]
        else:
            self.epsilon = self.get_epsilon_exponential(episode_number)
            if np.random.uniform() > self.epsilon:
                next_action = self.next_action(self.Q[state])
            else:
                next_action = np.argmax(np.random.uniform(0, 1, size=self.actions))

        return next_action
    
    def get_epsilon_linear(self, k, n):
        res = (n - k) / n
        return np.amax([res, self.epsilon_lower_bound])

    def get_epsilon_exponential(self, n):
        res = 1 / (n + 1)
        return np.amax([res, self.epsilon_lower_bound])

    def next_action(self, state):
        """
        It chooses the best action given the current state.
        Paramteres:
        state -> array of possible actions in the current state.
        """
        max_value = np.amax(state)
        max_indexes = np.arange(len(state))[state == max_value]
        np.random.shuffle(max_indexes)
        return max_indexes[0]

    