
# coding: utf-8

from random import randint

class RingBuffer:
    def __init__(self, max_buffer_size):
            self.max_buffer_size = max_buffer_size
            self.current_index = 0
            self.buffer = [None] * self.max_buffer_size
            self.stored_elements = 0

    def append(self, item):
            self.buffer[self.current_index] = item
            self.current_index = (self.current_index + 1) % self.max_buffer_size
            if self.stored_elements <= self.max_buffer_size:
                    self.stored_elements += 1

    def random_pick(self, n_elem):
            picks = []
            for _ in range(n_elem):
                    rand_index = randint(0, min(self.stored_elements, self.max_buffer_size) - 1)
                    picks.append(self.buffer[rand_index])
            return picks

    def mean(self):
            acc = 0
            for i in range(min(self.stored_elements, 100)):
                    acc += self.buffer[i]
            return acc/self.stored_elements


# In[ ]:


import random as ran
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard


def huber_loss(a, b, in_keras=True):
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
                 use_ddqn=False, default_policy=False, model_filename=None, tb_dir="None",
                 epsilon=1, epsilon_lower_bound=0.1, epsilon_decay_function=lambda e: e - (0.9 / 1000000),
                 gamma=0.95, optimizer=RMSprop(0.00025), learn_thresh=50000,
                 update_rate=10000):
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
            if self.use_ddqn == False:
                if not end:
                    reward = reward + self.gamma * np.amax(self.evaluate_model.predict(new_state)[0])

                new_prediction = self.evaluate_model.predict(state)
                new_prediction[0][next_action] = reward
            else:
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
        return self.memory.random_pick(self.batch_size)

    def act(self, state):
        if np.random.uniform() > self.epsilon:
            # state = np.float32(state / 255) # TODO: generalisation
            prediction = self.evaluate_model.predict(state)
            next_action = np.argmax(prediction[0])
        else:
            next_action = np.argmax(np.random.uniform(0, 1, size=self.output_size))

        if self.total_steps > self.learn_thresh:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        self.total_steps += 1

        return next_action

    def memoise(self, t):
        if not self.default_policy:
            self.memory.append(t)

    def learn(self):
        if (self.total_steps > self.learn_thresh and
            (self.total_steps % self.update_rate) == 0 and not self.default_policy and
            self.use_ddqn == True):
            self.update_target_model()
            print("model updated, epsilon:", self.epsilon)
        if self.total_steps > self.learn_thresh and not self.default_policy and self.total_steps % 4 == 0:   
            self.replay()

    def save_model(self, filename):
        self.evaluate_model.save('%s.h5' % filename)
    
    def load_model(self, filename):
        return load_model('%s.h5' % filename, custom_objects={ 'huber_loss': huber_loss })


# In[ ]:


import numpy as np
from enum import Enum

class EnsemblerType(Enum):
    MAJOR_VOTING_BASED = 0
    AGGREGATION_BASED = 1
    TRUST_BASED = 2



class EnsemblerAgent:
    def __init__(self, output_size, agents, ensembler_type):
        self.agents = agents
        self.output_size = output_size
        self.ensembler_type = ensembler_type

        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            self.votes = np.zeros(self.output_size)
    
    def act(self, state):
        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            for agent in self.agents:
                self.votes[agent.act(state)] += 1
            action = np.argmax(self.votes)
            # print(self.votes)
            self.votes = np.zeros(self.output_size)

            return action

        else:
            return 0


# In[25]:


import os
import time
import gym
import numpy as np
import keras.optimizers 
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from tqdm import tqdm


def accuracy(results):
    """
    Evaluate the accuracy of results, considering victories and defeats.
    """
    return results[1] / (results[0] + results[1]) * 100


def experiment(n_episodes, default_policy=False, policy=None, render = False):
    res = [0, 0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    #env = env.unwrapped
    env.seed(91)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
        
    agent1 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model01", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent2 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model02", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent3 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model35", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent4 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model33", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    # agent5 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="model33", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent6 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi1", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    agent7 = DQNAgent(output_dim, None, use_ddqn=True, default_policy=True, model_filename="luigi2", epsilon=0.01, epsilon_lower_bound=0.01, learn_thresh=0)
    
    agents = [agent1, agent6, agent7]
    
    agent = EnsemblerAgent(output_dim, agents, EnsemblerType.MAJOR_VOTING_BASED)
    # agent = EnsemblerAgent(output_dim, agents, 0)
    
    for i_episode in tqdm(range(n_episodes), desc="Episode"):
        state = env.reset()
        cumulative_reward = 0

        state = np.reshape(state, [1, 2])
        
        for t in range(env._max_episode_steps):
            if (render):
                env.render()

            next_action = agent.act(state)                       
            new_state, reward, end, _ = env.step(next_action)

            reward = abs(new_state[0] - (-0.5)) # r in [0, 1]
            new_state = np.reshape(new_state, [1, 2])

            if end:
                if t == env._max_episode_steps - 1:
                    res[0] += 1
                else:
                    res[1] += 1
                    # print("ENTRATO!,", t, "steps")

                steps.append(t)
                break
            else:
                state = new_state
                cumulative_reward += reward
                
        cumulative_reward += reward
        scores.append(cumulative_reward)
    env.close()
    return {"results": np.array(res), "steps": np.array(steps), "scores": np.array(scores), "agent": agent }
    
# Training
res = experiment(500)
print("Accuracy:", accuracy(res["results"]), "Mean steps:", np.mean(res["steps"]), "Mean score:", np.mean(res["scores"]))
# res["agent"].save_model("ensembler01")

# Testing
#res2 = experiment(500, default_policy=True, policy="ensembler01")
#print("Testing accuracy: %s, Training mean score: %s" % (accuracy(res2["results"]), np.mean(res["scores"])))

# Rendering
#experiment(10, render=True, default_policy=True, policy="model1")

