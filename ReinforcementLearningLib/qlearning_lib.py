import numpy as np


class QLAgent:

    def __init__(self, shape, alpha=0.8, gamma=0.95, policy=None, epsilon=1,
                 epsilon_lower_bound=0.01, epsilon_decay_function=lambda e: e * 0.6):
        """
        Args:
            shape: a tuple that describes the state space tensor shape
            alpha: learning rate hyperparameter
            gamma: discount factor hyper parameter
            policy: numpy tensor test policy
            epsilon: annealing function upper bound
            epsilon_lower_bound: annealing function lower bound
            epsilon_decay_function: lambda annealing function 
        """
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.Q = np.zeros(shape)
        self.epsilon = epsilon
        self.epsilon_lower_bound = epsilon_lower_bound
        self.epsilon_decay_function = epsilon_decay_function
        self.policy = policy
        self.actions = shape[-1]
        self.discrete_state = True
        np.random.seed(91)

    def update_q(self, state, new_state, action, reward):
        """
        Apply Q-Learning update rule.

        Args:
            state: current state t
            new_state: next state t
            reward: reward
            action: current action
        """
        future_action = np.argmax(self.Q[new_state]) # Find the best action to perform at time t+1
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * self.Q[new_state][future_action])

    def act(self, state=None, return_prob_dist=False):
        """
        Return the action for current state.

        Args:
            state: current state 
            return_prob_dist: boolean for probability distribution used by ensemblers

        Returns:
            next_action: next action to perform
            prediction: probability distribution
        """
        if (self.policy is not None):
            next_action = self.policy[state]
        else:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

            # Annealing
            if np.random.uniform() > self.epsilon:
                prediction = self.Q[state]
                next_action = self.next_action(prediction)
            else:
                prediction = np.random.uniform(0, 1, size=self.actions)
                next_action = np.argmax(prediction)

        if not return_prob_dist:
            return next_action
        return next_action, prediction
    

    def next_action(self, state):
        """
        Choose the best action given the current state.

        Args:
            state: array of possible actions in the current state.

        Returns:
            max_indexes[0]: best action for current state
        """
        max_value = np.amax(state)
        max_indexes = np.arange(len(state))[state == max_value]
        np.random.shuffle(max_indexes)
        return max_indexes[0]
