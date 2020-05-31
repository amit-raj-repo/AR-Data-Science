# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:27:58 2020

@author: amit.sanghvi
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
# Decision Tree Algorithm
#===============================================================#

model = DecisionTreeClassifier(max_depth = 10,
                                   min_sample_split = 100,
                                   random_state = 3,
                                   min_sample_leaf = 50
                                   )
'''
Below is the list of important hyper parameter which helps in avoiding 
overfitting while using decision trees

Important Hyperparameters:
- max_depth
- min_samples_split
- random_state
- class_weight
'''

#Fitting the Model
model.fit(X_train, y_train)

#Getting the predictions
pred = model_clf.predict(X_test)

#Checking the accuracy
pred_acc = metrics.accuracy_score(y_test, pred)

#Checking the confusion metrics
conf_mat = confusion_matrix(y_train, pred)

'''
There are a bunch of features that are available after fitting the model:
- predict_proba
- get_depth
- feature_importance

checkout https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
'''
