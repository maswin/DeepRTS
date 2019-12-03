from keras.layers import Dense, Input, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.core import Dense
import random
import keras
import keras.backend as K
from keras.models import Sequential, Model, load_model
import numpy as np
from keras.callbacks import TensorBoard, History
import tensorflow as tf
from time import time
from collections import deque


class Actor_Critic():
    def __init__(self, load_network = False, load_weight = False, load_file = None):
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.0001
        self.discount_factor = 0.99
        self.num_actions = 4
        self.actor_policy, self.actor_predict = self.create_actor_network(load_network, load_weight, load_file)
        self.critic = self.create_critic_network()
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.value_memory = []
        self.next_state_memory = []
        self.actor_history = History()
        self.critic_history = History() 

    
    def create_actor_network(self, load_network = False, load_weight = False, load_file = None):
        
        if load_network is True:
            model = load_model(load_file)
            return (None, model)
        
        input = Input(shape = (363,))
        advantage = Input(shape = [1])
        dense1 = Dense(128, activation = 'relu')(input)
        dense2 = Dense(64, activation = 'relu')(dense1)
        prob_output = Dense(self.num_actions, activation = 'softmax')(dense2)
        opt = Adam(self.actor_learning_rate)

        def custom_loss(y_true, y_pred):
            clip_out = K.clip(y_pred,1e-8, 1-1e-8)
            log_lik = y_true * K.log(clip_out)
            return K.sum(-log_lik * advantage)

        policy_n = Model(inputs = [input, advantage], outputs = [prob_output])
        policy_n.compile(loss = custom_loss, optimizer=opt)
        predict_n = Model(inputs= [input], outputs = [prob_output])

        if load_weight is True:
            predict_n.load_weights(load_file)
            return (None, predict_n)

        return policy_n, predict_n
        
    def create_critic_network(self):
        model = Sequential()
        model.add(Dense(128, activation ='relu', input_dim = 363))
        # model.add(Dropout(0.15))
        model.add(Dense(64, activation ='relu'))
        # model.add(Dropout(0.15))
        model.add(Dense(1, activation = 'linear'))
        opt = Adam(self.critic_learning_rate)
        model.compile(loss = 'mean_squared_error', optimizer=opt, metrics = ['accuracy'])

        return model

    def actor_prediction(self, state):
        predicted_probs = self.actor_predict.predict(state)[0]  
        pred_action = np.random.choice(range(self.num_actions), p = predicted_probs)
        return pred_action

    def critic_prediction(self, state): 
        return (self.critic.predict(state))

    def predict_action(self, state):
        action = self.actor_prediction(state)
        return action
    
    def learn(self, state, action, reward, next_state, done):
        state_v = self.critic_prediction(state)
        next_state_v = self.critic_prediction(next_state)
        
        target = reward + self.discount_factor * next_state_v * (1- int(done))
        advantage = target - state_v

        actions = np.zeros((1,self.num_actions))
        actions[np.arange(1), action] = 1
    
        self.actor_policy.fit([state, advantage], actions, verbose=0, callbacks=[self.actor_history])
        loss = self.actor_history.history['loss'][0]
        
        f = open("./logs_ac/model_metrics_actor.csv",'a+')
        f.write(str(loss)+ "\n")
        f.close()

        self.critic.fit(state, target, verbose=0, callbacks=[self.critic_history])
        loss = self.critic_history.history['loss'][0]
        
        f = open("./logs_ac/model_metrics_critic.csv",'a+')
        f.write(str(loss)+ "\n")
        f.close()
        

    def save_model(self, iteration='1'):
        self.actor_predict.save_weights("./weight_store"+"/ac_weight_"+iteration+".h5")
        self.actor_predict.save("./model_store"+"/ac_model_"+iteration+".h5")

# def get_random_state():
#     return(np.random.rand(1,300))

# network_obj = Actor_Critic()
# s= get_random_state()
# print(network_obj.predict_action(s))
# s1 = get_random_state()
# r = 0.6
# a = 1
# done = False
# network_obj.learn(s,a,r,s1,done)
# # network_obj.remember(s,network_obj.predict_action(s),1)
# # s= get_random_state()
# # network_obj.remember(s,network_obj.predict_action(s),1)
# # s= get_random_state()
# # network_obj.remember(s,network_obj.predict_action(s),1)
# # s= get_random_state()
# # network_obj.remember(s,network_obj.predict_action(s),1)
# # s= get_random_state()
# # network_obj.remember(s,network_obj.predict_action(s),1)
# # s= get_random_state()
# # network_obj.remember(s,network_obj.predict_action(s),1)
# Referred to : https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code
