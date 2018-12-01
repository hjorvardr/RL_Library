from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
import numpy as np
import tensorflow as tf


class ACAgent:

    class Actor:

        def __init__(self, layers, tb_dir, default_policy=None):
            self.loss = "mean_squared_error"
            self.optimizer = Adam(lr=0.0001)
            # Tensorboard parameters
            self.tb_dir = tb_dir
            self.tb_step = 0
            self.tb_gather = 500
            if tb_dir is not None:
                self.tensorboard_actor = TensorBoard(log_dir='./Monitoring/%s/actor' % tb_dir, write_graph=False)
                print("Tensorboard Loaded! (log_dir: %s)" % self.tensorboard_actor.log_dir)

            if default_policy is None:
                self.model = Sequential()
                for l in layers:
                    self.model.add(l)
                self.model.compile(loss=self.loss, optimizer=self.optimizer)
            else:
                self.model = default_policy

        def learn(self, state, action, td_error):
            new_prediction = self.model.predict(state)
            new_prediction[0][action] = -td_error

            if (self.tb_step % self.tb_gather) == 0 and self.tb_dir is not None:
                self.model.fit(state, new_prediction, verbose=0, callbacks=[self.tensorboard_actor])
            else:
                self.model.fit(state, new_prediction, verbose=0)
            self.tb_step += 1


    class Critic:

        def __init__(self, layers, tb_dir, default_policy=None):
            self.optimizer = Adam(lr=0.001)
            self.gamma = 0.9
            self.tb_dir = tb_dir
            # Tensorboard parameters
            self.tb_step = 0
            self.tb_gather = 500

            if tb_dir is not None:
                self.tensorboard_critic = TensorBoard(log_dir='./Monitoring/%s/critic' % tb_dir, write_graph=False)
                print("Tensorboard Loaded! (log_dir: %s)" % self.tensorboard_critic.log_dir)

            if default_policy is None:
                self.model = Sequential()
                for l in layers:
                    self.model.add(l)
                self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
            else:
                self.model = default_policy

        def learn(self, state, new_state, reward):
            td_error = reward + self.gamma * self.model.predict(new_state)[0] - self.model.predict(state)[0]

            if (self.tb_step % self.tb_gather) == 0 and self.tb_dir is not None:
                self.model.fit(state, td_error, verbose=0, callbacks=[self.tensorboard_critic])
            else:
                self.model.fit(state, td_error, verbose=0)
            self.tb_step += 1

            return td_error

    def __init__(self, output_shape, actor_layers, critic_layers, default_policy=False, model_filename=None,
                tb_dir="tb_log", epsilon=1, epsilon_lower_bound=0.1, epsilon_decay_function=lambda e: e - (0.9 / 950000),
                gamma=0.95):
        self.output_shape = output_shape
        self.default_policy = default_policy

        # Exploration/Exploitation parameters
        self.epsilon = epsilon
        self.epsilon_decay_function = epsilon_decay_function
        self.epsilon_lower_bound = epsilon_lower_bound

        # Learning parameters
        self.gamma = gamma
        self.loss = 'mean_squared_error'

        # Model init
        if not default_policy:
            self.actor_model = self.Actor(actor_layers, tb_dir)
            self.critic_model = self.Critic(critic_layers, tb_dir)
        else:
            self.actor_net, self.critic_net = self.load_model(model_filename)
            self.actor_model = self.Actor(None, tb_dir, self.actor_net)
            self.critic_model = self.Critic(None, tb_dir, self.critic_net)

    def act(self, state):
        if np.random.uniform() > self.epsilon:
            prediction = self.actor_model.model.predict(state)
            next_action = np.argmax(prediction[0])
        else:
            next_action = np.argmax(np.random.uniform(0, 1, size=self.output_shape))

        self.epsilon = self.epsilon_decay_function(self.epsilon)
        self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        return next_action

    def learn(self, state, action, new_state, reward, end):
        if not self.default_policy:
            td_error = self.critic_model.learn(state, new_state, reward)
            self.actor_model.learn(state, action, td_error)
    
    def save_model(self, filename):
        self.actor_model.model.save('%s-actor.h5' % filename)
        self.critic_model.model.save('%s-critic.h5' % filename)

    def load_model(self, filename):
        return load_model('%s-actor.h5' % filename), load_model('%s-critic.h5' % filename)
