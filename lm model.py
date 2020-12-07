# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:00:47 2019

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

weather_data = pd.read_csv('WeatherHistory.csv')
weather_data.head(3)

All=weather_data.describe(include='all')
Cat=weather_data.describe(include=['O'])
Cor=weather_data.corr()
data_set=weather_data.iloc[:,[0,2,3,4,5,8]]
Cor_s=data_set.corr()

sns.boxplot(x=data_set["Humidity"])
data_set_clean = data_set[data_set["Humidity"]>0.20]
sns.boxplot(x=data_set_clean["Humidity"])

sns.boxplot(x=data_set["Temperature (C)"])
data_set_clean = data_set[data_set["Temperature (C)"]>-20]
sns.boxplot(x=data_set_clean["Temperature (C)"])

sns.boxplot(x=data_set["Apparent Temperature (C)"])
data_set_clean = data_set[data_set["Apparent Temperature (C)"]>-26]
sns.boxplot(x=data_set_clean["Apparent Temperature (C)"])

y= data_set_clean.iloc[:,[2]]
X= data_set_clean.iloc[:,[1,3,4]]

X1= pd.get_dummies(X, columns =['Precip Type'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
lm.intercept_

y_pred = lm.predict(X_test)
plt.scatter(y_test,y_pred)
sns.distplot((y_test-y_pred),bins=50);

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import statsmodels.api as sm
model = sm.OLS(y_pred,X_test).fit()
model.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]

New_X_train=X_train.drop(["Precip Type_snow"],axis = 1,inplace = True)
New_X_test=X_test.drop(["Precip Type_snow"],axis = 1,inplace = True)

lm.fit(X_train, y_train)
lm.coef_
lm.intercept_

y_pred = lm.predict(X_test)
plt.scatter(y_test,y_pred)
sns.distplot((y_test-y_pred),bins=50);

import statsmodels.api as sm
model = sm.OLS(y_pred,X_test).fit()
model.summary()



X_train.columns


















