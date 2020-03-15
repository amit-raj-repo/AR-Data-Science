import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import xgboost


parameters_xgb_1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}

parameters_xgb_2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

parameters_xgb_3 = {
 'n_estimators':range(200,800,100)
}

parameters_xgb_4 = {
 'reg_lambda':[0.2, 0.5, 0.7, 0.9]
}

parameters_xgb_5 = {
 'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
}

def gridSearchFunction(model, model_params):
    
    grid_search_xgb = GridSearchCV(estimator = model,
                           param_grid = model_params,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = 4)
    grid_search = grid_search_xgb.fit(X_train, y_train)
    return grid_search

xgModel = xgboost.XGBClassifier()
test = gridSearchFunction(xgModel, parameters_xgb_4)


def model_tuning():
    """
    Please Note: This function requires a lot of resources and time to run
    
    This functions is used to tune the hyperparameters for XG Boost.
    It starts with min_depth and min_child_weight then moves to gamma, n_estimators,
    reg_lambda and finally to learning_rate. The tuned parameters are finally stored
    in dict_
    """
    
    param_array = [parameters_xgb_1, parameters_xgb_2, parameters_xgb_3, parameters_xgb_4, parameters_xgb_5]
    #param_dict = ['n_estimators', 'max_depth']
    xgModel = xgboost.XGBClassifier()    
    dict_ = {}
    
    
    for i in range(0,5):
        model_params = param_array[i]
        grid_search_op = gridSearchFunction(xgModel, model_params)
        dict_.update(grid_search_op.best_params_)
        xgModel = xgboost.XGBClassifier(**dict_)
        
    finalModel = xgModel.fit(X_train, y_train)
    predxGb = finalModel.predict(X_test)
    pred_acc_xgb_f = metrics.accuracy_score(y_test, predxGb)
    cm_xgb_f = confusion_matrix(y_test, predxGb)

    predxGb = finalModel.predict(X_train)
    pred_acc_xgb_f = metrics.accuracy_score(y_train, predxGb)
    cm_xgb_train_f = confusion_matrix(y_train, predxGb)  
