import numpy as np
from qlearning_lib import QLAgent


class SARSAAgent(QLAgent):

    def __init__(self, shape, alpha=0.8, gamma=0.95, policy=None):
        super().__init__(shape, alpha, gamma, policy)
        self.current_policy = None

    def extract_policy(self):
        self.current_policy = [self.next_action(state) for state in self.Q] 

    def update_q(self, state, new_state, action, reward):
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
        next_action = self.current_policy[new_state]
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * self.Q[new_state][next_action])

    def act(self, state, episode_number):
        if (self.policy is not None):
            next_action = self.policy[state]
        else:
            self.epsilon = self.get_epsilon_exponential(episode_number)
            if np.random.uniform() > self.epsilon:
                next_action = self.current_policy[state]
            else:
                next_action = np.argmax(np.random.uniform(0, 1, size=self.actions))

        return next_action