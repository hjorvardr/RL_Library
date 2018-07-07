import random as ran
from collections import deque
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard


class DQNAgent:

    def __init__(self, input_size, output_size, layers, memory_size=3000, batch_size=32,
                 use_ddqn=False, default_policy=False, model_filename=None, tb_dir="tb_log"):
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.use_ddqn = use_ddqn
        self.default_policy = default_policy
        # Tensorboard parameters
        self.tb_step = 0
        self.tb_gather = 500
        self.tensorboard = TensorBoard(log_dir='./Monitoring/%s' % tb_dir,
                                       histogram_freq=0, write_graph=False)
        print("Tensorboard Loaded! (log_dir: %s)" % self.tensorboard.log_dir)
        # Exploration/Exploitation parameters
        self.epsilon = 1
        self.epsilon_decay_rate = 0.95
        self.epsilon_lower_bound = 0.01
        self.learn_step = 0
        self.total_steps = 0
        # Learning parameters
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.loss = 'mean_squared_error'
        self.optimizer = RMSprop(self.learning_rate)
        self.batch_size = batch_size
        self.learn_thresh = 1000 # Number of steps from which the network starts learning
        self.update_rate = 300

        if self.default_policy:
            self.evaluate_model = self.load_model(model_filename)
        else:
            self.evaluate_model = self.build_model(layers)

            if self.use_ddqn:
                self.target_model = self.build_model(layers)
        self.evaluate_model.summary()

    def build_model(self, layers):
        model = Sequential()
        for l in layers:
            model.add(l)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.evaluate_model.get_weights())

    def replay(self):
        pick = self.random_pick()
        for state, next_action, reward, new_state, end in pick:
            if not end:
                reward = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])

            new_prediction = self.target_model.predict(state)
            new_prediction[0][next_action] = reward
            
            if (self.tb_step % self.tb_gather) == 0:
                self.evaluate_model.fit(state, new_prediction, verbose=0, callbacks=[self.tensorboard])
            else:
                self.evaluate_model.fit(state, new_prediction, verbose=0)
            self.tb_step += 1

    def random_pick(self):
        return ran.sample(self.memory, self.batch_size)

    def act(self, state):
        if (self.default_policy):
            prediction = self.evaluate_model.predict(state)
            return np.argmax(prediction[0])
        else:
            if np.random.uniform() > self.epsilon:
                prediction = self.evaluate_model.predict(state)
                next_action = np.argmax(prediction[0])
            else:
                next_action = np.argmax(np.random.uniform(0, 1, size=self.output_size))

        if self.total_steps > self.learn_thresh:
            self.epsilon = self.epsilon * self.epsilon_decay_rate
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        self.total_steps += 1

        return next_action

    def memoise(self, t):
        if not self.default_policy:
            self.memory.append(t)

    def learn(self):
        if (self.total_steps > self.learn_thresh and
            (self.total_steps % self.update_rate) == 0 and not self.default_policy):
            self.update_target_model()
        if self.total_steps > self.learn_thresh and not self.default_policy:   
            self.replay()

    def save_model(self, filename):
        self.evaluate_model.save('%s.h5' % filename)
    
    def load_model(self, filename):
        return load_model('%s.h5' % filename)