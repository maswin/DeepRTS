from keras.preprocessing.image import save_img
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.core import Dense
import random
from keras.models import Sequential
import numpy as np
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard


class DoubleDeepQNetwork(object):

    def __init__(self, saved_weights=False,saved_state = False, one_model=None,twin_model = None,):
        self.PER = False
        self.train_tick = 0
        # Memory = 20000 before training starts
        # update every 8000
        self.tick = 0
        self.predict_tick = 0

        # 0.00025 0.0001 0.0005 0.01
        self.alpha = 0.0003
        self.tau = 0.05

        if saved_weights is True:
            self.one_model = self.create_network(True,one_model)
            self.twin_model = self.create_network(True,twin_model)

        elif(saved_state is True):
            self.one_model = load_model(one_model)
            self.twin_model = load_model(twin_model)

        else:
            self.one_model = self.create_network()
            self.twin_model = self.create_network()
        # 0.95 , 0.98 , 0.99
        self.discount_factor = 0.99


        #### MAYBE CAN GET MORE STABLE


        # 1e-4 , 4e-4 1e-6
        self.epsilon = 0.98
        self.batch_size = 100

        self.memory = []
        self.prioritized_replay = dict()
        self.mem_size = 0

        self.tensorboard1 = TensorBoard(log_dir="logs1/{}".format(time()))
        self.tensorboard2 = TensorBoard(log_dir="logs2/{}".format(time()))

    def load_model(self,model_path,which_model):
        if (which_model == 1):
            self.one_model = load_model(model_path)
        else:
            self.twin_model = load_model(model_path)





    def create_network(self, weightfile_bool=False,weightfile = None):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(3, 3), activation='relu', input_shape=(320, 320, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(self.alpha, clipvalue=1.0), metrics=['accuracy'])

        if weightfile_bool is True:
            model.load_weights(weightfile)
        return model

    def remember(self, state, action, reward, next_state, finished):
        if(self.mem_size > 10000):
            if (random.randint(2) == 1):
                #add to memory

                key = random.randint(0, 9999)
                self.memory[key] = (state, action, reward, next_state, finished)
        else:
            self.mem_size += 1
            self.memory.append((state, action, reward, next_state, finished))






    def replay_new(self):
        # 2000 , 8000 , 10000 , 32
        if self.mem_size <70:
            pass

        else:


            # Change to this

            #random_keys = random.sample(range(1,self.mem_size), 64)
            # for i in random_keys:
                #state, action, reward, next_state, finished = self.dict_memory[i]  # why 0
                #end_result = reward if finished is True else (
                #        self.discount_factor * np.max(self.twin_model.predict(next_state)[0]))
                #target = self.twin_model.predict(state)
                #target[0][action] = end_result
                # target[0][np.argmax(action)] = end_result
                #self.one_model.fit(state, target, epochs=1, verbose=0)


            minibatch = random.sample(self.memory,64)
            for state, action, reward, next_state, finished in minibatch:  # why 0
                end_result = reward if finished is True else (
                        self.discount_factor * np.max(self.twin_model.predict(next_state)[0]))
                target = self.twin_model.predict(state)
                target[0][action] = end_result
                # target[0][np.argmax(action)] = end_result
                self.one_model.fit(state, target, epochs=1, verbose=2,callbacks = [self.tensorboard2])

    def immediate_update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.discount_factor * np.amax(self.twin_model.predict(next_state)[0])
        reward_pred = self.one_model.predict(state)
        reward_pred[0][action] = target
        self.one_model.fit(state, reward_pred, epochs=1, verbose=2,callbacks = [self.tensorboard1])

    def target_train(self):
        # 1/5th frequency of replay_new
        weights = self.one_model.get_weights()
        target_weights = self.twin_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.twin_model.set_weights(target_weights)

    def predict_action(self, state):
        # self.memory.append(state)
        self.predict_tick +=1
        rand_prob = np.power(self.epsilon,self.predict_tick)

        if(rand_prob <0.01):
            rand_prob = 0.01


        if(random.random() <= rand_prob):
            if(random.random() >= 0.5):
                return 0
            else:
                return 1



        return np.argmax(self.one_model.predict(state)[0])

    def train(self, state, action, reward, next_state, done):
        self.train_tick+=1
        self.remember(state, action, reward, next_state, done)
        self.immediate_update(state, action, reward, next_state, done)
        if(self.train_tick %100 == 0 ):
           self.replay_new()

        if(self.train_tick % 50 ==0):
             self.target_train()






    def get_summary(self):
        return self.one_model.summary()

    def save_model(self,iteration='1'):
        self.one_model.save_weights("model_one"+iteration+".h5")
        self.one_model.save("model_one_state"+iteration+".h5")

        self.twin_model.save_weights("model_two" + iteration + ".h5")
        self.twin_model.save("model_two_stte" + iteration + ".h5")




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
