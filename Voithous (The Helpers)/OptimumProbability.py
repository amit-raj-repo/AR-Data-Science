import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler


def optimum_probability(fitted_model, x_train, y_train):
    
    """
    Function returns a dataframe with important values such as accuracy, precision, TPR etc.
    It will be helpful in finding the optimum probability to draw decision boundries.
    
    Parameters:
        fitted_model = requires a fitted model e.g. LogisticRegression().fit(X data, y data)
        x_test & y_test = requires the complete test data 
    """
    
    pred = fitted_model.predict_proba(x_train)
    test_class = pd.DataFrame(y_train, columns = ["original_value"])
    predicted_probability = pd.DataFrame(pred, columns = ["prob_0", "prob_1"])
    test_predict_probability = pd.concat([test_class, predicted_probability], axis = 1)
    
    probability = np.arange(0,1.01,0.01)

    value_list = []
    for prob in probability:
        test_predict_probability['new_predicted'] = np.where(test_predict_probability["prob_1"] >= prob, 1, 0)
        tp = np.where(np.logical_and(test_predict_probability["original_value"] == 1, test_predict_probability['new_predicted'] == 1), 1, 0).sum()
        tn = np.where(np.logical_and(test_predict_probability["original_value"] == 0, test_predict_probability['new_predicted'] == 0), 1, 0).sum()
        fp = np.where(np.logical_and(test_predict_probability["original_value"] == 0, test_predict_probability['new_predicted'] == 1), 1, 0).sum()
        fn = np.where(np.logical_and(test_predict_probability["original_value"] == 1, test_predict_probability['new_predicted'] == 0), 1, 0).sum()
        
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        fpr = fp/(fp+tn)
        fnr = fn/(fn+tp)
        value_list.append([prob, accuracy, precision, tpr, fpr, tnr, fnr])
        
    logistic_dataframe = pd.DataFrame(value_list, 
                                      columns = ["probability","accuracy", 
                                                 "precision", "tpr", "fpr", 
                                                 "tnr", "fnr"])
    
    return logistic_dataframe
    
#Usage
    
logistic_dataframe = optimum_probability(xgbModel,X_train, y_train)

logistic_dataframe.plot.line(y=["tpr", "accuracy", 'precision'], figsize=(12,10),lw=2)

print(logistic_dataframe)

pd.DataFrame.save(logistic_dataframe)
logistic_dataframe.to_csv("probability_cutoff")
