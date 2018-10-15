import random as ran
from collections import deque
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply


class ACAgent:
    class Actor:
        def __init__(self, layers):
            self.loss = "mean_squared_error"
            self.optimizer = Adam(lr=0.001)
            self.model = Sequential()
            for l in layers:
                self.model.add(l)
            self.model.compile(loss=self.loss, optimizer=self.optimizer)

            print("Actor model:")
            self.model.summary()

        def learn(self, state, action, td_error):
            prediction = self.model.predict(state)
            log_prob = np.log(prediction[0][action])

            target = log_prob * td_error

            new_prediction = self.model.predict(state)
            new_prediction[0][action] = -target
            self.model.fit(state, new_prediction, verbose=0)


    class Critic:
        def __init__(self, layers):
            self.optimizer = Adam(lr=0.01)
            self.gamma = 0.99
            self.model = Sequential()
            for l in layers:
                self.model.add(l)
            self.model.compile(loss=self.loss, optimizer=self.optimizer)

            print("Critic model:")
            self.model.summary()


        def loss(self, curr_value, pred_value):
            return np.square(pred_value)

        
        def learn(self, state, new_state, reward):
            td_error = reward + self.gamma * self.model.predict(new_state)[0] - self.model.predict(state)[0]

            self.model.fit(state, td_error, verbose=0)

            return td_error

    def __init__(self, output_shape, actor_layers, critic_layers, default_policy=False, model_filename=None,
                tb_dir="tb_log", epsilon=1, epsilon_lower_bound=0.1, epsilon_decay_function=lambda e: e - (0.9 / 950000),
                 gamma=0.95, learn_thresh=50000, update_rate=10000):
        self.output_shape = output_shape
        self.default_policy = default_policy

        # Tensorboard parameters
        self.tb_step = 0
        self.tb_gather = 500
        if tb_dir is not None:
            self.tensorboard = TensorBoard(log_dir='./Monitoring/%s' % tb_dir, write_graph=False)
            print("Tensorboard Loaded! (log_dir: %s)" % self.tensorboard.log_dir)

        # Exploration/Exploitation parameters
        self.epsilon = epsilon
        self.epsilon_decay_function = epsilon_decay_function
        self.epsilon_lower_bound = epsilon_lower_bound

        # Learning parameters
        self.total_steps = 0
        self.gamma = gamma
        self.loss = 'mean_squared_error'
        self.learn_thresh = learn_thresh # Number of steps from which the network starts learning
        self.update_rate = update_rate

        # Model init
        self.actor_model = self.Actor(actor_layers)
        self.critic_model = self.Critic(critic_layers)

    def act(self, state):
        if np.random.uniform() > self.epsilon:
            prediction = self.actor_model.model.predict(state)
            next_action = np.argmax(prediction[0])
        else:
            next_action = np.argmax(np.random.uniform(0, 1, size=self.output_shape))

        if self.total_steps > self.learn_thresh:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        self.total_steps += 1

        return next_action

    def learn(self, state, action, new_state, reward, end):
        if self.total_steps > self.learn_thresh:
            td_error = self.critic_model.learn(state, new_state, reward)
            self.actor_model.learn(state, action, td_error)
    