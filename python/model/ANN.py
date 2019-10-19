from keras.preprocessing.image import save_img
from keras.layers import Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.core import Dense
import random
from keras.models import Sequential
import numpy as np

class ANN:

    def __init__ (self):
        self.memory = []
        self.tick = 0 
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.model = self.create_network()
        self.epsilon = np.power(0.97, self.tick)


    def create_network(self):
        model = Sequential()
        model.add(Dense(128, kernel_initializer="uniform", activation='relu', input_dim = 300))
        model.add(Dropout(0.15))
        model.add(Dense(64, kernel_initializer="uniform", activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(2, kernel_initializer="uniform", activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        return model


    def remember(self, state, action, reward, next_state, finished):
        self.memory.append((state, action, reward, next_state, finished))


    def replay_new(self):
        if len(self.memory) > 10000:
            minibatch = random.sample(self.memory, 64)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, finished in minibatch: 
            end_result = reward if finished is True else (
                    self.discount_factor * np.max(self.model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = end_result
            self.model.fit(state, target, epochs=1, verbose=0)

    def immediate_update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
        reward_pred = self.model.predict(state)
        reward_pred[0][action] = target
        self.model.fit(state, reward_pred, epochs=1, verbose=0)

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.immediate_update(state, action, reward, next_state, done)
        if len(self.memory) > 2:
            self.replay_new()

    def predict_action(self, state):
        self.tick = self.tick + 1
        return np.argmax(self.model.predict(state)[0])

    def get_summary(self):
        return self.model.summary()
