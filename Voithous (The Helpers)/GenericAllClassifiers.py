# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:34:08 2019

@author: amit.sanghvi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

# load the datasets
df = pd.read_csv("DataSet.csv").dropna(axis = 0)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


#split datasets into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('# of training data : ', X_train.shape[0])
print('# of test data : ', X_test.shape[0])
      
# Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

      
model_acc = {}

# KNeighborsClassifier
model_clf = KNeighborsClassifier()
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["KNeighborsClassifier"] = pred_acc


#DecisionTreeClassifier
model_clf = DecisionTreeClassifier()
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["DecisionTreeClassifier"] = pred_acc


#RandomForestClassifier
model_clf = RandomForestClassifier()
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["RandomForestClassifier"] = pred_acc

#RandomForestClassifier
model_clf = RandomForestClassifier()
model_clf.fit(X_train_scaled, y_train)
pred = model_clf.predict(X_test_scaled)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["RandomForestClassifierScaled"] = pred_acc


#LogisticRegression
model_clf = LogisticRegression()
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred > 0.3)
model_acc["LogisticRegression"] = pred_acc


#SGDClassifier(max_iter=100)
model_clf = SGDClassifier(max_iter=100)
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["SGDClassifier"] = pred_acc


#MultinomialNB
model_clf = MultinomialNB()
model_clf.fit(X_train, y_train)
pred = model_clf.predict(X_test)
pred_acc = metrics.accuracy_score(y_test, pred)
model_acc["MultinomialNB"] = pred_acc


for key, value in model_acc.items():
    print(key, value)
