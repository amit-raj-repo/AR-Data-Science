# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:33:14 2020

@author: amit.sanghvi
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# load the datasets
df = pd.read_csv("DataToBeImputed.csv")

def imputeFunction(df):
    data_dict = {}
    data_dict_t4 = {}
    identity_flag_values = df.identity_flag_new.unique()
    
    for i in identity_flag_values:
        
        df_token4 = df[df.identity_flag_new == i].iloc[:, 0].values
        df_all = df[df.identity_flag_new == i].iloc[:, 1:].values
        df_col = df[df.identity_flag_new == i].iloc[:, 1:]
        
        print(df_all.shape)
        
        imp = IterativeImputer(estimator = KNeighborsRegressor(n_neighbors=15), max_iter=20, random_state=0, verbose = 1)
        df_filled = imp.fit_transform(df_all)
        data_dict[i] = df_filled
        data_dict_t4[i] = df_token4
     
    
    df_0 = pd.DataFrame(data_dict[0], columns= df_col.columns).dropna(axis = 0)
    df_0['token4'] = data_dict_t4[0]
    
    df_1 = pd.DataFrame(data_dict[1], columns= df_col.columns).dropna(axis = 0)
    df_1['token4'] = data_dict_t4[1]
    
    df_2 = pd.DataFrame(data_dict[2], columns= df_col.columns).dropna(axis = 0)
    df_2['token4'] = data_dict_t4[2]
    
    df_appended = df_0.append(df_1).append(df_2)
   
    
    return df_appended


dfFinal = imputeFunction(df)

dfFinal.to_csv("dfFinal")
