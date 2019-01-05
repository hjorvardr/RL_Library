from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
import numpy as np
import tensorflow as tf
from ring_buffer import RingBuffer


def huber_loss(a, b, in_keras=True):
    """
    Apply Huber loss function

    Args:
        a: target value
        b: predicted value
        in_keras: use keras backend

    Returns:
        Huber loss value
    """
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class DQNAgent:

    def __init__(self, output_size, layers, memory_size=3000, batch_size=32,
                 use_ddqn=False, default_policy=False, model_filename=None, tb_dir="tb_log",
                 epsilon=1, epsilon_lower_bound=0.1, epsilon_decay_function=lambda e: e - (0.9 / 1000000),
                 gamma=0.95, optimizer=RMSprop(0.00025), learn_thresh=50000,
                 update_rate=10000):
        """
        Args:
            output_size: number of actions
            layers: list of Keras layers
            memory_size: size of replay memory
            batch_size: size of batch for replay memory
            use_ddqn: boolean for choosing between DQN/DDQN
            default_policy: boolean for loading a model from a file
            model_filename: name of file to load
            tb_dir: directory for tensorboard logging
            epsilon: annealing function upper bound
            epsilon_lower_bound: annealing function lower bound
            epsilon_decay_function: lambda annealing function 
            gamma: discount factor hyper parameter
            optimizer: Keras optimiser
            learn_thresh: number of steps to perform without learning
            update_rate: number of steps between network-target weights copy
        """
        self.output_size = output_size
        self.memory = RingBuffer(memory_size)
        self.use_ddqn = use_ddqn
        self.default_policy = default_policy
        # Tensorboard parameters
        self.tb_step = 0
        self.tb_gather = 500
        self.tb_dir = tb_dir
        if tb_dir is not None:
            self.tensorboard = TensorBoard(log_dir='./Monitoring/%s' % tb_dir, write_graph=False)
            print("Tensorboard Loaded! (log_dir: %s)" % self.tensorboard.log_dir)
        # Exploration/Exploitation parameters
        self.epsilon = epsilon
        self.epsilon_decay_function = epsilon_decay_function
        self.epsilon_lower_bound = epsilon_lower_bound
        self.total_steps = 0
        # Learning parameters
        self.gamma = gamma
        self.loss = huber_loss #'mean_squared_error'
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learn_thresh = learn_thresh # Number of steps from which the network starts learning
        self.update_rate = update_rate
        self.discrete_state = False

        if self.default_policy:
            self.evaluate_model = self.load_model(model_filename)
        else:
            self.evaluate_model = self.build_model(layers)

            if self.use_ddqn:
                self.target_model = self.build_model(layers)

    def build_model(self, layers):
        """
        Build a Neural Network.

        Args:
            layers: list of Keras NN layers

        Returns:
            model: compiled model with embedded loss and optimiser
        """
        model = Sequential()
        for l in layers:
            model.add(l)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def update_target_model(self):
        """
        Set target net weights to evaluation net weights.
        """
        self.target_model.set_weights(self.evaluate_model.get_weights())

    def replay(self):
        """
        Perform DQN learning phase through experience replay.
        """
        pick = self.random_pick()
        for state, next_action, reward, new_state, end in pick:
        # for state, next_action, reward, frame, end in pick:
            # state = np.float32(state / 255) # for CNN learning
            # frame = np.float32(frame / 255) # for CNN learning
            # new_state = np.append(frame, state[:, :, :, :3], axis=3) # for CNN learning

            # Simple DQN case
            if self.use_ddqn == False:
                if not end:
                    reward = reward + self.gamma * np.amax(self.evaluate_model.predict(new_state)[0])

                new_prediction = self.evaluate_model.predict(state)
                new_prediction[0][next_action] = reward
            else:
                # Double DQN case
                if not end:
                    action = np.argmax(self.evaluate_model.predict(new_state)[0])
                    reward = reward + self.gamma * self.target_model.predict(new_state)[0][action]

                new_prediction = self.target_model.predict(state)
                new_prediction[0][next_action] = reward

            if (self.tb_step % self.tb_gather) == 0 and self.tb_dir is not None:
                self.evaluate_model.fit(state, new_prediction, verbose=0, callbacks=[self.tensorboard])
            else:
                self.evaluate_model.fit(state, new_prediction, verbose=0)
            self.tb_step += 1

    def random_pick(self):
        """
        Pick a random set of elements from replay memory of size self.batch_size.

        Returns:
            set of random elements from memory
        """
        return self.memory.random_pick(self.batch_size)

    def act(self, state, return_prob_dist=False):
        """
        Return the action for current state.

        Args:
            state: current state t
            return_prob_dist: boolean for probability distribution used by ensemblers

        Returns:
            next_action: next action to perform
            prediction: probability distribution
        """
        # Annealing
        if np.random.uniform() > self.epsilon:
            # state = np.float32(state / 255) # for CNN learning
            prediction = self.evaluate_model.predict(state)[0]
            next_action = np.argmax(prediction)
        else:
            prediction = np.random.uniform(0, 1, size=self.output_size)
            next_action = np.argmax(prediction)

        # Start decaying after self.learn_thresh steps
        if self.total_steps > self.learn_thresh:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        self.total_steps += 1

        if not return_prob_dist:
            return next_action
        return next_action, prediction

    def memoise(self, t):
        """
        Store tuple to replay memory.

        Args:
            t: element to store
        """
        if not self.default_policy:
            self.memory.append(t)

    def learn(self):
        """
        Perform the learning phase.
        """
        # Start target model update after self.learn_thresh steps
        if (self.total_steps > self.learn_thresh and
            (self.total_steps % self.update_rate) == 0 and not self.default_policy and
            self.use_ddqn == True):
            self.update_target_model()
        # Start learning after self.learn_thresh steps
        if self.total_steps > self.learn_thresh and not self.default_policy and self.total_steps % 4 == 0:   
            self.replay()

    def save_model(self, filename):
        """
        Serialise the model to .h5 file.

        Args:
            filename
        """
        self.evaluate_model.save('%s.h5' % filename)
    
    def load_model(self, filename):
        """
        Load model from .h5 file

        Args:
            filename

        Returns:
            model
        """
        return load_model('%s.h5' % filename, custom_objects={ 'huber_loss': huber_loss })
