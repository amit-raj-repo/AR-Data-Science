# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:57:04 2020

@author: amit.sanghvi
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix


df = pd.read_csv("df_you_want_to_input.csv").dropna(axis = 0)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

# Defining a PCA Solver function, please note the whiten = True is of great importance here
# Read more at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

def PcaSolver(df, n_comp):
    pca = PCA(n_components = n_comp, whiten=True, random_state=3)
    transformedDf = pca.fit_transform(df)
    return transformedDf, pca

#Running PCA solver to get the PC array
# Here we notice that n_comp = 10, Just take a random value of n_comp <= Total columns in the data to start with 
# Run pcaVal.explained_variance_ratio_ to get the variance contribution by PCs
# Sum it up and when it's around 99% thats the number of PCs you need to consider
# In my case the optimum number of n_comp was 10

pcaDf, pcaVal = PcaSolver(X, 10)

#/*=========================================================================*/
# Running ADA boost using the PCA output  
#/*=========================================================================*/

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(pcaDf, y, test_size=0.25, random_state=3)

#You can tune the hyper parameters as required please check Impute function in helpers
adaDef = AdaBoostClassifier(n_estimators=300, learning_rate = 0.1,
                            random_state = 3)
adaDef.fit(X_train, y_train)

predAda = adaDef.predict(X_test)

pred_acc_ada = metrics.accuracy_score(y_test, predAda)

cm_ada = confusion_matrix(y_test, predAda)



