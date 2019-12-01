#from sklearn import datasets
import catboost as cb
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

import pandas as pd


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import random
import pickle


class NPC_History:

  def __init__(self ):
      self.BAYES_State = []
      self._Action = []
      self.num_obs = 0
      self.do_once_flag = True


  def Add_Observation(self,state,action):
      self.BAYES_Action.append(action)
      self.BAYES_State.append(state)
      self.num_obs+=1

class NPC_BAYES:

  def __init__(self,NPC_type = 'Follower',load_model= False, data = None, val = None):
      self.load_model = load_model
      self.X = None
      self.Y = None
      self.clf = None
      self.eval_clf = None
      if self.load_model is True:
        self.X = np.load(data)
        self.Y = np.load(val)
        self.clf = GaussianNB()
        self.clf.fit(self.X,self.Y)










  def train_(self,state_t,action_t):
    self.X = state_t
    #print(X)
    self.y = action_t

    self.clf = GaussianNB()
    self.clf.fit(self.X, self.Y)





  # Cross Entropy
  def eval_train(self):

      self.eval_clf = GaussianNB()
      self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)
      self.eval_clf.fit(self.X_train,self.Y_train)
      Y_pred = self.eval_clf.predict(self.X_test)
      print("Number of mislabeled points out of a total %d points : %d"%(len(self.Y_test), (self.Y_test != Y_pred).sum()))

  def get_pred(self,state_t):
      self.preds = self.clf.predict_proba(state_t)
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
    le = preprocessing.LabelEncoder()
    le.fit([0, 5, 1, 3, 4, 2, 6])
    print(le.classes_)
    a = NPC_BAYES('follower',False)
    iris = pd.read_csv('seeds_dataset.txt',delim_whitespace=True)
    X = iris.iloc[:,[0,1,2,3,4,5,6]]
    print(X.head)
    y = iris.iloc[:,[7]]


    print(y.head)

    a.train_(X,y)
    a.eval_train()



#print('got here')

