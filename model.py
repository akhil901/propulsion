#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv('propulsion.csv')

df.shape

df.drop(['Unnamed: 0'],axis=1,inplace=True)

x=df.drop(['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.'],axis=1)
y_compressor=df['GT Compressor decay state coefficient.']
y_turbine = df['GT Turbine decay state coefficient.']


x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y_compressor,test_size=0.3,random_state=42)
x_train2,x_test2,y_train2,y_test2 = train_test_split(x,y_turbine,test_size=0.3,random_state=42)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(min_samples_split= 2,min_samples_leaf= 1,max_features= 'sqrt',
                                                                  bootstrap=False,random_state=42)

def model_evl(train_features,train_labels,test_features,test_labels,name,model=rfr):
    model.fit(train_features,train_labels)
    y_pred = model.predict(test_features)
    R2_score = r2_score(test_labels,y_pred)
    rmse = np.sqrt(mean_squared_error(test_labels, y_pred))
    print("the r2_score on {} is {}:".format(name,R2_score))
    print("the rmse_score on {} is {}:".format(name,rmse))
    print(70*'*')
    
    
model_evl(x_train1,y_train1,x_test1,y_test1,'GT Compressor decay state coefficient')
model_evl(x_train2,y_train2,x_test2,y_test2,'GT Compressor turbine state coefficient')

