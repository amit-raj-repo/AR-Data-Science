# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:19:50 2020
@author: amit.sanghvi
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost
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
# Fitting XGBoost to the Training set
#============================================================================#
xgBoostDef = xgboost.XGBClassifier(base_score = 0.4, n_estimators= 300, max_depth = 9, 
                                   min_child_weight=1, gamma=0.3, 
                                   reg_lambda=0.5, learning_rate = 0.1)

xgbModel = xgBoostDef.fit(X_train, y_train)
predxGb = xgbModel.predict(X_test)
pred_acc_xgb = metrics.accuracy_score(y_test, predxGb)
cm_xgb = confusion_matrix(y_test, predxGb)

predxGb = xgbModel.predict(X_train)
pred_acc_xgb_train = metrics.accuracy_score(y_train, predxGb)
cm_xgb_train = confusion_matrix(y_train, predxGb)

#============================================================================#
#HyperParameter Tuning
#============================================================================#

xgBoostDef = xgboost.XGBClassifier()

parameters_xgb = [{'max_depth':[4,6,8,10],
               'n_estimators': [200, 400, 500, 600],
               'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
               'reg_lambda': [0.2, 0.5, 0.7, 0.9],
               'base_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
               'min_child_weight': [0.5,1,2]
               }
]

grid_search_xgb = GridSearchCV(estimator = xgBoostDef,
                           param_grid = parameters_xgb,
                           scoring = 'recall',
                           cv = 5,
                           n_jobs = -1)
grid_searchxgb_output = grid_search_xgb.fit(X_train, y_train)
