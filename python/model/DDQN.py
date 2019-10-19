from keras.preprocessing.image import save_img
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.core import Dense
import random
from keras.models import Sequential
import numpy as np


class DoubleDeepQNetwork(object):

    def __init__(self, weights=None):
        self.PER = False

        # Memory = 20000 before training starts
        # update every 8000
        self.tick = 0

        # 0.00025 0.0001 0.0005 0.01
        self.alpha = 0.0003
        self.tau = 0.05
        if weights is not None:
            self.weights = weights
        # 0.95 , 0.98 , 0.99
        self.discount_factor = 0.99
        self.one_model = self.create_network()

        #### MAYBE CAN GET MORE STABLE
        self.twin_model = self.create_network()

        # 1e-4 , 4e-4 1e-6
        self.epsilon = np.power(0.97, self.tick)
        self.batch_size = 100

        self.memory = []
        self.prioritized_replay = dict()

    def create_network(self, weightfile=False):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(3, 3), activation='relu', input_shape=(320, 320, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(self.alpha, clipvalue=1.0), metrics=['accuracy'])

        if weightfile is True:
            model.load_weights(self.weights)
        return model

    def remember(self, state, action, reward, next_state, finished):
        self.memory.append((state, action, reward, next_state, finished))

    def replay_new(self):
        # 2000 , 8000 , 10000 , 32
        if len(self.memory) > 10000:
            minibatch = random.sample(self.memory, 1000)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, finished in minibatch:  # why 0
            end_result = reward if finished is True else (
                    self.discount_factor * np.max(self.twin_model.predict(next_state)[0]))
            target = self.twin_model.predict(state)
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

    def target_train(self):
        # 1/5th frequency of replay_new
        weights = self.one_model.get_weights()
        target_weights = self.twin_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.twin_model.set_weights(target_weights)

    def predict_action(self, state):
        # self.memory.append(state)
        return np.argmax(self.one_model.predict(state)[0])

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.immediate_update(state, action, reward, next_state, done)
        if len(self.memory) > 2:
            self.replay_new()

    def get_summary(self):
        return self.one_model.summary()


# Testing purpose
if __name__ == "__main__":
    rgb = np.load('/Users/dhruvmathew/PycharmProjects/snake/tempoutfile.npy')

    k = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).reshape((320, 320))
    k = np.expand_dims(k, axis=2)
    np.save("game_img", k)

    save_img('grey_game.jpeg', k)

    print(k.shape)
    k = np.expand_dims(k, axis=0)

    A = DoubleDeepQNetwork()
    A.create_network()

    ans = A.one_model.predict(k)

    print(A.one_model.summary())

    print(ans.shape)
    print(ans)

    print(ans[0])
