# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:27:58 2020

@author: amit.sanghvi
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Data Import

df = pd.read_csv("df_you_want_to_input.csv").dropna(axis = 0)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

#Splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25,
                                                    random_state=3)
#===============================================================#
#Decision Tree Algorithm
#===============================================================#

model = RandomForestClassifier(n_estimators=200,
                                   max_depth = 10,
                                   min_sample_split = 100,
                                   random_state = 3,
                                   min_sample_leaf = 50,
                                   bootstrap=True,
                                   verbose=1                                   
                                   )
'''
Below is the list of important hyper parameter which helps in avoiding 
overfitting while using Random Forest

Important Hyperparameters:
- n_estimators
- max_depth
- min_samples_split
- random_state
- class_weight
- bootstrap
- verbose
'''

#Fitting the Model
model.fit(X_train, y_train)

#Getting the predictions
pred = model.predict(X_test)

#Checking the accuracy
pred_acc = metrics.accuracy_score(y_test, pred)

#Checking the confusion metrics
conf_mat = confusion_matrix(y_train, pred)

'''
There are a bunch of features that are available after fitting the model:
- predict_proba
- get_depth
- feature_importance
checkout https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details
'''
