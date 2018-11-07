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
                 use_ddqn=False, default_policy=False, model_filename=None, tb_dir=None,
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
        #self.evaluate_model.summary()

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
        # for state, next_action, reward, frame, end in pick:
            # state = np.float32(state / 255) # TODO: generalisation
            # frame = np.float32(frame / 255) # TODO: generalisation
            # new_state = np.append(frame, state[:, :, :, :3], axis=3) # TODO: generalisation
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
            prediction = self.evaluate_model.predict(state)[0]
            next_action = np.argmax(prediction)
        else:
            prediction = np.random.uniform(0, 1, size=self.output_size)
            next_action = np.argmax(prediction)

        if self.total_steps > self.learn_thresh:
            self.epsilon = self.epsilon_decay_function(self.epsilon)
            self.epsilon = np.amax([self.epsilon, self.epsilon_lower_bound])

        self.total_steps += 1

        return next_action, prediction

    def memoise(self, t):
        if not self.default_policy:
            self.memory.append(t)

    def learn(self):
        if (self.total_steps > self.learn_thresh and
            (self.total_steps % self.update_rate) == 0 and not self.default_policy and
            self.use_ddqn == True):
            self.update_target_model()
        if self.total_steps > self.learn_thresh and not self.default_policy and self.total_steps % 4 == 0:   
            self.replay()

    def save_model(self, filename):
        self.evaluate_model.save('%s.h5' % filename)
    
    def load_model(self, filename):
        return load_model('%s.h5' % filename, custom_objects={ 'huber_loss': huber_loss })


# In[23]:


import os
import time
import gym
import numpy as np
import keras.optimizers 
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from tqdm import tqdm


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(91)
tf.set_random_seed(91)

def experiment(n_episodes, default_policy=False, policy=None, render=False):
    res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
    scores = [] # Cumulative rewards
    steps = [] # Steps per episode
    
    env = gym.make('MountainCar-v0')
    env.seed(91)
    n_states = 150

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    layer1 = Dense(15, input_dim=input_dim, activation='relu')
    layer2 = Dense(output_dim)

    
    if (default_policy):
        agentE = QLAgent([n_states, n_states, env.action_space.n], policy=policy, epsilon=0.01, epsilon_lower_bound=0.01)
    else:
        agent0 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01, optimizer=keras.optimizers.Adam(0.001))
        agent1 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01, optimizer=keras.optimizers.Adam(0.001))
        agent2 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01, optimizer=keras.optimizers.Adam(0.001))
        agent3 = DQNAgent(output_dim, [layer1, layer2], use_ddqn=True, learn_thresh=1000, update_rate=300, epsilon_decay_function=lambda e: e * 0.95, epsilon_lower_bound=0.01, optimizer=keras.optimizers.Adam(0.001))
        agents = [agent1, agent2, agent3]
        agentE = EnsemblerAgent(env.action_space.n, agents, EnsemblerType.RANK_VOTING_BASED)

    evaluate = False
    
    for i_episode in tqdm(range(n_episodes + 1), desc="Episode"):
        state_original = env.reset()
        
        state = obs_to_state(env, state_original, n_states)
        cumulative_reward = 0
        
        if i_episode > 0 and i_episode % 30 == 0:
            evaluate = True
            
        state = np.reshape(state, [1, 2])
        
        if evaluate == False:
            
        
            for t in range(env._max_episode_steps):
                if (render):
                    env.render()

                next_action = agentE.act(state)                       
                state_original, reward, end, _ = env.step(next_action)

                
                new_state = np.reshape(state_original, [1, 2])
                                
                #agent0.memoise((state, next_action, reward, new_state, end))
                agent1.memoise((state, next_action, reward + 0.4 * state_original[0], new_state, end))
                agent2.memoise((state, next_action, reward + 0.5 * np.sin(3 * state_original[0]), new_state, end))
                agent3.memoise((state, next_action, reward + 0.9 * (state_original[1] * state_original[1]), new_state, end))
                
                if end:
                    if t == env._max_episode_steps - 1:
                        res[0] += 1
                    else:
                        res[1] += 1
                        print("ENTRATO!,", t, "steps")

                    steps.append(t)
                    break
                else:
                    state = new_state
                    cumulative_reward += reward

                agent0.learn()
                agent1.learn()
                agent2.learn()
                agent3.learn()
                
            cumulative_reward += reward
            scores.append(cumulative_reward)
            env.close()

        else:
            evaluate = False
            eval_res = [0,0] # array of results accumulator: {[0]: Loss, [1]: Victory}
            eval_scores = [] # Cumulative rewards
            eval_steps = [] # Steps per episode

            for i_episode in range(500):
                if (render):
                    env.render()

                next_action = agentE.act(state)                       
                state_original, reward, end, _ = env.step(next_action)

                
                new_state = np.reshape(state_original, [1, 2])
                
                if end:
                    if t == env._max_episode_steps - 1:
                        res[0] += 1
                    else:
                        res[1] += 1
                        print("ENTRATO!,", t, "steps")

                    eval_steps.append(t)
                    break
                else:
                    state = new_state
                    cumulative_reward += reward
            cumulative_reward += reward
            eval_scores.append(cumulative_reward)
            env.close()

            testing_accuracy = accuracy(np.array(res))
            testing_mean_steps = np.array(eval_steps).mean()
            testing_mean_score = np.array(eval_scores).mean()
            print("\nTraining episodes:", len(steps), "Training mean score:", np.array(steps).mean(),             "Training mean steps", np.array(scores).mean(), "\nAccuracy:", testing_accuracy, "Test mean score:", testing_mean_score, "Test mean steps:", testing_mean_steps)

    return 0 # {"results": np.array(res), "steps": np.array(eval_steps), "scores": np.array(eval_scores), "Q": agent0.Q}


# Training
train_res = experiment(2500)


# In[ ]:




