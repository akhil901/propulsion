#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


df = pd.read_csv('propulsion.csv')

df.shape

df.drop(['Unnamed: 0'],axis=1,inplace=True)

x=df.drop(['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.'],axis=1)
y=df[['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.']]

from sklearn.externals import joblib
joblib_file = "Pickle_RFR_Model.pkl" 
joblib_RFR_model = joblib.load(joblib_file)

y_pred = joblib_RFR_model.predict(x)
R2_score = r2_score(y,y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("the r2_score on test_data is {}:".format(R2_score))
print("the rmse_score on test_data is {}:".format(rmse))



