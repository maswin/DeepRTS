from keras.preprocessing.image import save_img
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.core import Dense
import random
from keras.models import Sequential
import numpy as np

class ANN:

    def __init__ (self):
        self.memory = [];


    def create_network(self):
        model = Sequential()
        model.add(Dense(output_dim = 120, activation='relu', input_dim =(300,)))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim = 120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim = 2, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        return model


    def remember(self, state, action, reward, next_state, finished):
        self.memory.append((state, action, reward, next_state, finished))


    def replay_new(self, model):
        # 2000 , 8000 , 10000 , 32
        if len(self.memory) > 10000:
            minibatch = random.sample(self.memory, 1000)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, finished in minibatch:  # why 0
            end_result = reward if finished is True else (
                    self.discount_factor * np.max(model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = end_result
            # target[0][np.argmax(action)] = end_result
            self.one_model.fit(state, target, epochs=1, verbose=0)

    def immediate_update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.twin_model.predict(next_state)[0])
        reward_pred = self.one_model.predict(state)
        reward_pred[0][action] = target
        self.one_model.fit(state, reward_pred, epochs=1, verbose=0)

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.immediate_update(state, action, reward, next_state, done)
        if len(self.memory) > 2:
            self.replay_new()

