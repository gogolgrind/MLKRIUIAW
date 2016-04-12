# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:13:28 2016

@author: Kostya S.
"""
from sklearn import  datasets
from sklearn import cross_validation as crv
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
import scipy as sp

class Model():
    X = None
    y = None
    clf = None
    X_train, X_test, y_train, y_test,y_pred = [None  for i in range(5)]
    def __init__(self,model_name = 'KNN',params = []):
        if model_name == 'KNN':
            if params != []:
                self.clf = KNN(n_neighbors = params[0])
            else:
                self.clf = KNN()
        elif model_name == 'DT':
            if params != []:
                self.clf = DT(max_depth = params[0])
            else:
                self.clf = DT()
        else:
            self.clf = SVC()
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
        
    def get_train_data(self):
        yt = self.y_train[:, sp.newaxis]
        return sp.hstack((self.X_train,yt))
        
    def get_class_report(self):
        yt = self.y_test[:, sp.newaxis]
        yp = self.y_pred[:, sp.newaxis]
        return  sp.hstack((yt,yp))
        
    def get_test_data(self):
        yt = self.y_test[:, sp.newaxis]
        return sp.hstack((self.X_test,yt))
        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = crv.train_test_split(self.X, self.y, 
                                                            test_size=0.4, random_state = 11)
    def train_pred(self):
        self.clf.fit(self.X_train,self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
    
    def get_accuracy(self):
        a = accuracy_score(self.y_test,self.y_pred)
        return a            
        
