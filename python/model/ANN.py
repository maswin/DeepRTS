from keras.layers import Dropout
from keras.optimizers import Adam
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

class ANN:

    def __init__ (self, load_model = False, load_weight = False, load_file = None):
        self.memory = []
        self.tick = 1 
        self.learning_rate = 0.0004
        self.discount_factor = 0.99
        self.model = self.create_network(load_model, load_weight, load_file)
        self.epsilon = np.power(0.97, self.tick)
        self.MAX_MEMORY_LENGTH = 10000
        self.SAMPLE_SIZE = 32
        self.replay_history = History()
        #self.tensorboard1 = TensorBoard(log_dir = "logs\log_inter_ann")
        #self.tensorboard2 = TensorBoard(log_dir = "logs\log_replay_ann")


    def create_network(self, load_model = False, load_weight = False, load_file = None):

        if load_model is True:
            model = load_model(load_file)
            return model

        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim = 300))
        model.add(Dropout(0.15))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(2, activation='linear'))
        opt = Adam(self.learning_rate)
        model.compile(loss = tf.keras.losses.Huber(), optimizer=opt, metrics = ['accuracy'])

        if load_weight is True:
            model.load_weights(load_file)

        return model


    def remember(self, state, action, reward, next_state, finished):
        if(len(self.memory) < self.MAX_MEMORY_LENGTH):
            self.memory.append((state, action, reward, next_state, finished))
        else:
            ind = np.random.randint(self.MAX_MEMORY_LENGTH)
            self.memory[ind] = (state, action, reward, next_state, finished)


    def replay_new(self):
        if(len(self.memory) <= self.SAMPLE_SIZE):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.SAMPLE_SIZE)
        
        states = []
        targets = []
        for state, action, reward, next_state, finished in minibatch: 
            end_result = reward if finished is True else (
                    self.discount_factor * np.max(self.model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = end_result
            states.append(state.reshape((300,)))
            targets.append(target.reshape(2,))
            
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.replay_history])
        avg_loss = self.replay_history.history['loss'][0]
        
        #avg_loss = avg_loss / self.SAMPLE_SIZE
        f = open("./logs_ann/model_metrics_replay.csv",'a+')
        f.write(str(avg_loss)+ "\n")
        f.close()

    def immediate_update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
        reward_pred = self.model.predict(state)
        reward_pred[0][action] = target
        self.model.fit(state, reward_pred, epochs=1, verbose=0 )


    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.immediate_update(state, action, reward, next_state, done)
        if(self.tick % 100 == 0):
            self.replay_new()

    def save_model(self, iteration='1'):
        self.model.save_weights("./weight_store"+"/weight"+iteration+".h5")
        self.model.save("./model_store"+"/model"+iteration+".h5")

    def predict_action(self, state):
        self.tick = self.tick + 1
        return np.argmax(self.model.predict(state)[0])

    def get_summary(self):
        return self.model.summary()
