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
# Fitting GradientBoost to the Training set
#============================================================================#

GBdef = GradientBoostingClassifier(n_estimators= 500,
                                   min_samples_leaf = 100, max_depth = 12,
                                   n_iter_no_change = 50, random_state= 1)
GbModel = GBdef.fit(X_train, y_train)

predGb = GbModel.predict(X_test)
pred_acc_gb = metrics.accuracy_score(y_test, predGb)
#print(pred_acc_gb)

cm_gb = confusion_matrix(y_test, predGb)

predGb = GbModel.predict(X_train)
cm_gb_train = confusion_matrix(y_train, predGb)

#The set of hyperparamerts used are after running the tuner. Please check out Model tuning in helper function.
