from keras.optimizers import Adam
from keras.layers import Dense, Input, Activation
import random
import keras.backend as K
from keras.models import Sequential, Model, load_model
import numpy as np
import tensorflow as tf
from time import time
from keras.callbacks import History
from collections import deque


class PG():
    def __init__(self, load_network = False, load_weight = False, load_file = None):
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.num_actions = 4
        self.curr_disc_rewards = None
        self.policy_n, self.predict_n = self.create_network(load_network, load_weight, load_file)
    
    def create_network(self, load_network = False, load_weight = False, load_file = None):

        if load_network is True:
            model = load_model(load_file)
            return (None, model)

        input = Input(shape = (363,))
        disc_rewards = Input(shape = [1])
        dense1 = Dense(128, activation = 'relu')(input)
        dense2 = Dense(64, activation = 'relu')(dense1)
        prob_output = Dense(self.num_actions, activation = 'softmax')(dense2)
        opt = Adam(self.learning_rate)

        def custom_loss(y_true, y_pred):
            clip_out = K.clip(y_pred,1e-8, 1-1e-8)
            log_lik = y_true * K.log(clip_out)
            return K.sum(-log_lik * disc_rewards)

        policy_n = Model(inputs = [input, disc_rewards], outputs = [prob_output])
        policy_n.compile(loss = custom_loss, optimizer=opt)
        predict_n = Model(inputs= [input], outputs = [prob_output])

        if load_weight is True:
            predict_n.load_weights(load_file)
            return (None, predict_n)

        return policy_n, predict_n

    def predict_action(self, state):
        predicted_probs = self.predict_n.predict(state)[0]  
        pred_action = np.random.choice(range(self.num_actions), p = predicted_probs)
        return pred_action

    def remember(self, state, action, reward):
        self.state_memory.append(state.reshape((363,)))
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def update_policy(self):
        state_mem = np.array(self.state_memory)
        action_mem = np.array(self.action_memory)
        reward_mem = np.array(self.reward_memory)

        actions = np.zeros((len(action_mem),self.num_actions))
        actions[np.arange(len(action_mem)), action_mem] = 1
 
        disc_rewards = np.zeros_like(reward_mem)

        for t in range(len(reward_mem)):
            Gt = 0 
            pw = 0
            for r in reward_mem[t:]:
                Gt = Gt + (self.discount_factor ** pw) * r
                pw = pw + 1
            disc_rewards[t] = Gt
        
        #scale rewards for numerical stability - baseline
        mean = disc_rewards.mean()
        std_dev = disc_rewards.std()
        
        norm_disc_rewards = (disc_rewards-mean)/std_dev

        # Train on the network
        cost = self.policy_n.train_on_batch([state_mem, norm_disc_rewards],actions)

        # Reset the memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        f = open("./logs_pg/model_metrics_pg.csv",'a+')
        f.write(str(cost)+ "\n")
        f.close()

        return cost
    
    def save_model(self, iteration='1'):
        self.predict_n.save_weights("./weight_store"+"/pg_weight_"+iteration+".h5")
        self.predict_n.save("./model_store"+"/pg_model_"+iteration+".h5")
'''
def get_random_state():
    return(np.random.rand(1,300))

network_obj = PG()
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)
s= get_random_state()
network_obj.remember(s,network_obj.predict_action(s),1)

print(network_obj.update_policy())
'''