#from sklearn import datasets
import catboost as cb
import numpy as np

import pandas as pd


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import random
import pickle


class NPC:
  def __init__(self,NPC_type,load_model= False,model_type = "CatClassifier.bin", epsilon =0.1):
      self.load_model = load_model
      if (self.load_model is True):
          self.model = CatBoostClassifier()
          self.model.load_model(model_type)

      else:
        model = CatBoostClassifier(
          custom_loss=['MultiClass'],
          iterations= 300,
          eval_metric='Accuracy',
          use_best_model= True,
          random_seed=42,
          logging_level='Silent',
          od_type =  'Iter',
          od_wait=  50
        )




        self.model = model
      #self.clf = xgb.XGBClassifier()
      self.epsilon = epsilon
      #self.booster = xgb.Booster()



      # State can either be DQN state going through autoencoder OR
      #MimicA state = Health,No_Enemies,RewardValue,Resources available,Dist. closest enemy base,Dist closest enemy to player, Last 5 Actions



      self.NPC_type = NPC_type


  def load_model(self,model_type):
      self.model.load_model(model_type)


  def train_(self,state_t,action_t):
    self.X = state_t
    #print(X)
    self.y = action_t

    self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.y, test_size=0.2)

    self.model.fit(
        self.X_train, self.Y_train,
        eval_set=(self.X_test, self.Y_test),
        #     logging_level='Verbose',  # you can uncomment this for text output
        #plot=True
    )





  # Cross Entropy
  def eval_train(self):

      cv_params = self.model.get_params()
      cv_params.update({
          'loss_function': 'Logloss'
      })
      cv_data = cv(
          Pool(self.X, self.y),
          cv_params,
          #plot=True
      )
      print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
          np.max(cv_data['test-Accuracy-mean']),
          cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
          np.argmax(cv_data['test-Accuracy-mean'])
      ))

      print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))

  def get_pred(self,state_t):
      self.preds = self.model.predict(state_t)
      a = random.uniform(0, 1)
      if (self.NPC_type == 'follower'):
          if(a> self.epsilon):
              self.best_preds = np.asarray([np.argmax(line) for line in self.preds])
          else:
              self.best_preds = np.asarray([np.argmin(line) for line in self.preds])

      elif (self.NPC_type == 'complementary'):
          if (a > self.epsilon):
            self.best_preds = np.asarray([np.argmin(line) for line in self.preds])

          else:
              self.best_preds = np.asarray([np.argmax(line) for line in self.preds])


  def find_best_params(self):
      pass





  def save_model(self):
      self.model.save_model('CatClassifier.bin')


def test():

    a = NPC('follower',False)
    iris = pd.read_csv('seeds_dataset.txt',delim_whitespace=True)
    X = iris.iloc[:,[0,1,2,3,4,5,6]]
    print(X.head)
    y = iris.iloc[:,[7]]


    print(y.head)

    a.train_(X,y)
    a.eval_train()



#print('got here')
