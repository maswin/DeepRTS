# Refrred to https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb
from keras.layers import Dropout
from keras.optimizers import Adam, RMSProp, SGD
from keras.layers.core import Dense
import random
import keras
from keras.models import load_model
from keras.models import Sequential
import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf
from time import time
from keras.callbacks import History
from collections import deque

class ANN:

    def __init__ (self, load_network = False, load_weight = False, load_file = None):
        
        memory_len = 10000
        self.PER = False
        self.tick = 1 
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        self.tau = 1
        self.num_actions = 4
        self.model = self.create_network(load_network, load_weight, load_file)
        self.target_model = self.create_network(False, False, False)
        self.epsilon = np.power(0.97, self.tick)
        self.MAX_MEMORY_LENGTH = memory_len
        self.SAMPLE_SIZE = 32
        self.memory = deque(maxlen = memory_len)
        self.priorities = deque(maxlen = memory_len)
        self.priority_scale = 1.0
        self.priority_offset = 0.1
        self.replay_history = History()

        #self.tensorboard1 = TensorBoard(log_dir = "logs\log_inter_ann")
        #self.tensorboard2 = TensorBoard(log_dir = "logs\log_replay_ann"


    def create_network(self, load_network = False, load_weight = False, load_file = None):

        if load_network is True:
            model = load_model(load_file)
            return model

        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim = 363))
        model.add(Dropout(0.15))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(self.num_actions, activation='linear'))
        opt = SGD(self.learning_rate, decay = 1e-6)
        model.compile(loss = tf.keras.losses.Huber(), optimizer=opt, metrics = ['accuracy'])

        if load_weight is True:
            model.load_weights(load_file)

        return model


    def remember(self, state, action, reward, next_state, finished):
        # if(len(self.memory) < self.MAX_MEMORY_LENGTH):
        self.memory.append((state, action, reward, next_state, finished))
        self.priorities.append(max(self.priorities, default=1))
        # else:
        #     ind = np.random.randint(self.MAX_MEMORY_LENGTH)
        #     self.memory[ind] = (state, action, reward, next_state, finished)

    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities) ** self.priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_sample_weights(self, probabilities):
        weights = 1/len(self.memory) * 1/probabilities
        weights_normalized = weights / max(weights)
        return weights_normalized

    def set_priorities(self, indices, errors):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + self.priority_offset

    def replay_new_PER(self):
        sample_size = min(len(self.memory), self.SAMPLE_SIZE)
        sample_probs = self.get_probabilities()
        sample_indices = random.choices(range(len(self.memory)), k=sample_size, weights=sample_probs)
        minibatch = np.array(self.memory)[sample_indices]
        sample_weights = (self.get_sample_weights(sample_probs[sample_indices])) ** (1-self.epsilon)

        errors = []
        for i in range(len(sample_indices)):
            state, action, reward, next_state, finished = minibatch[i] 
            end_result = reward  if finished is True else (
                    reward + self.discount_factor * np.max(self.target_model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = end_result
            self.model.fit(state, target, sample_weight = np.array([sample_weights[i]]), epochs=1, verbose=0, callbacks=[self.replay_history])
            avg_loss = self.replay_history.history['loss'][0]
            errors.append(avg_loss)
        
        self.set_priorities(sample_indices, errors)
        
        avg_loss = sum(errors)/sample_size        
        f = open("./logs_ann/model_metrics_PER.csv",'a+')
        f.write(str(avg_loss)+ "\n")
        f.close()


    def replay_new(self):

        sample_size = min(len(self.memory), self.SAMPLE_SIZE)
        minibatch = random.sample(self.memory, sample_size)
        
        states = []
        targets = []
        for state, action, reward, next_state, finished in minibatch: 
            end_result = reward  if finished is True else (
                    reward + self.discount_factor * np.max(self.target_model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = end_result
            self.model.fit(state, target, epochs=1, verbose=0, callbacks=[self.replay_history])
            avg_loss += self.replay_history.history['loss'][0]
            #states.append(state.reshape((363,)))
            #argets.append(target.reshape(self.num_actions,))
            
        #states = np.array(states)
        #targets = np.array(targets)
        #self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.replay_history])
        #loss = self.replay_history.history['loss'][0]
        avg_loss = avg_loss/sample_size
        f = open("./logs_ann/model_metrics_ER.csv",'a+')
        f.write(str(loss)+ "\n")
        f.close()

    def immediate_update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
        reward_pred = self.model.predict(state)
        reward_pred[0][action] = target
        self.model.fit(state, reward_pred, epochs=1, verbose=0)

    def transfer_weights(self):
        online_weights = self.model.get_weights()
        old_target_weights = self.target_model.get_weights()
        new_target_weights = []
        for i in range(len(old_target_weights)):
            new_target_weights.append(online_weights[i] * self.tau + old_target_weights[i] * (1 - self.tau))
        self.target_model.set_weights(new_target_weights)

    def set_PER(self, tf):
        self.PER = tf

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        #self.immediate_update(state, action, reward, next_state, done)
        if(self.PER):
            self.replay_new_PER()
        else:
            self.replay_new()
        # if(self.tick % 100 == 0):
        #     self.replay_new()

        # if(self.tick % 1000 == 0):
        #     self.transfer_weights()

    def save_model(self, iteration='1'):
        ex = "PER" if self.PER else "ER"
        self.model.save_weights("./weight_store" + "/ann_" + ex + "_weight_"+ iteration+".h5")
        self.model.save("./model_store" + "/ann_" + ex + "_model_" + iteration+".h5")

    def predict_action(self, state):
        self.tick = self.tick + 1
        return np.argmax(self.model.predict(state)[0])

    def get_summary(self):
        return self.model.summary()
