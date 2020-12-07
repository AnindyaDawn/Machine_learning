# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:56:57 2019

@author: LENOVO
"""

import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 

dataset = pd.read_csv('data.csv')
print(dataset.shape)
print(list(dataset.columns))

dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
dataset.isnull().sum()

sns.countplot(x='diagnosis',data=dataset, palette='hls')

sns.boxplot(y="radius_mean", data=dataset)
sns.heatmap(dataset.corr())

X = dataset.iloc[:,1:]
y = dataset.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
X_train.shape

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features





