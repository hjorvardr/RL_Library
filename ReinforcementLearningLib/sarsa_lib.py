import numpy as np
from qlearning_lib import QLAgent


class SARSAAgent(QLAgent):

    def __init__(self, shape, alpha=0.8, gamma=0.95, policy=None, epsilon=1,
                 epsilon_lower_bound=0.01, epsilon_decay_function=lambda e: e * 0.6, update_rate=100):
        """
        Args:
            shape: a tuple that describes the state space tensor shape
            alpha: learning rate hyperparameter
            gamma: discount factor hyper parameter
            policy: numpy tensor test policy
            epsilon: annealing function upper bound
            epsilon_lower_bound: annealing function lower bound
            epsilon_decay_function: lambda annealing function 
            update_rate: number of steps between policy generation operations
        """
        super().__init__(shape, alpha, gamma, policy, epsilon, epsilon_lower_bound,
        epsilon_decay_function)
        self.current_policy = None
        if policy is not None:
            self.current_policy = policy
        self.shape = shape
        self.update_rate = update_rate
        self.Q_target = None
        self.total_episodes = 0

    def extract_policy(self):
        """
        Generate the current policy.
        """
        if (self.total_episodes % self.update_rate) == 0:
            policy_shape = self.shape
            policy_shape = policy_shape[:-1]
            self.current_policy = np.zeros(policy_shape, dtype=int)
            for idx, _ in np.ndenumerate(self.current_policy):
                self.current_policy[idx] = self.next_action(self.Q[idx])
            self.Q_target = self.Q
        self.total_episodes += 1

    def update_q(self, state, new_state, action, reward):
        """
        Apply Q-Learning update rule.

        Args:
            state: current state t
            new_state: next state t
            reward: reward
            action: current action
        """
        next_action = self.current_policy[new_state]
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * self.Q[new_state][next_action])

    def act(self, state, return_prob_dist=False): # TODO: controllare episode_number
        """
        Return the action for current state.

        Args:
            state: current state
            return_prob_dist: boolean for probability distribution used by ensemblers

        Returns:
            next_action: next action to perform
            self.Q_target[state]: probability distribution
        """
        if (self.policy is not None):
            next_action = self.policy[state]
        else:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

            # Annealing
            if np.random.uniform() > self.epsilon:
                next_action = self.current_policy[state]
            else:
                next_action = np.argmax(np.random.uniform(0, 1, size=self.actions))

        if not return_prob_dist:
            return next_action
        return next_action, self.Q_target[state]
