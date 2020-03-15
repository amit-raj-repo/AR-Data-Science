# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:19:50 2020

@author: amit.sanghvi
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# load the datasets
df = pd.read_csv("dataSetName.csv").dropna(axis = 0)

desc = df.describe()

#Convert the data into numpy array
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#split datasets into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

#============================================================================#
# Fitting AdaBoost to the Training set
#============================================================================#

dt = DecisionTreeClassifier(max_depth=1)

adaDef = AdaBoostClassifier(n_estimators= 100, random_state= 1)
adaModel = adaDef.fit(X_train, y_train)

pred = adaModel.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
print(pred_acc)

cm_ada_test = confusion_matrix(y_test, predGb)

predAdaTrain = adaModel.predict(X_train)
cm_ada_train = confusion_matrix(y_train, predAdaTrain)


#============================================================================#
# After checking the accuracy, run the hyperparameter tuner to get the best
# set of parameters
#============================================================================#

parameters = [{'n_estimators': [50, 100, 200, 300, 400, 500, 600]}, 
              {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]}]

grid_search = GridSearchCV(estimator = adaDef,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
