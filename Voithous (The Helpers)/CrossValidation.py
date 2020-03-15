# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:45:56 2019

@author: amit.sanghvi

Notebook can be used to perform K Fold validation on the data
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

# load the datasets
df = pd.read_csv("DataSet.csv").dropna(axis = 0)

#Convert the data into numpy array
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# define the object of the classifier type, note: you can define any classifier with n hyperparameters

clf_lr = LogisticRegression()
clf_dt = DecisionTreeClassifier()

# number of samples in the complete dataset
n_samples = X.shape[0]

def crossValidation(model, splits, test_data_size = 0.2, scoring_parameters = ["accuracy"]):
    """
    Following function can be used to perform K fold validation by defining 
    different hyperparameters. This function returs the dictionary containing
    values of different scoring parameters
    
    Parameters:
        model - Object of the type of classifier you want to fit,
                e.g. LogisticRegression() or DecisionTreeClassifier()
        
        splits - This defines the K of K fold validation, i.e. the number of 
                 bins you want to split the data and number of corss validations
                 you want to perform
        
        test_data_size - % size of test dataset it can take any value between 0 and 1
                         default value is 20% i.e. 0.2
        
        scoring_parameters - There are a variety of parameters o which you can judge the model
                             e.g. precision, recall. By default the value is accuracy
    
    Resources for reference:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        https://scikit-learn.org/stable/modules/cross_validation.html
        
    """
    cv = ShuffleSplit(n_splits=splits, test_size=test_data_size, random_state=0)
    score = cross_validate(model, X, y, cv=cv, scoring = scoring_parameters)
    return score

def crossValidationOutput(output):
    """
    This function is used to display the values of the output dictionary from
    crossValidation() function
    """
    for key, value in output.items():
        print(key, value)

#Logistic Regression Cross Validation
logisticValidation = crossValidation(clf_lr, 5, 0.2)
crossValidationOutput(logisticValidation)

#Decision Tree Cross Validation
decisionTreeValidation = crossValidation(clf_dt, 5, 0.2,["accuracy", "precision", "recall"])
crossValidationOutput(decisionTreeValidation)

