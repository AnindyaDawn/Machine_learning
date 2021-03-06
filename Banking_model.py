# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:15:41 2019

@author: LENOVO
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('Banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

#Barplot for the dependent variable
sns.countplot(x='y',data=data, palette='hls')

#Check the missing values
data.isnull().sum()

#Customer job distribution
sns.countplot(y="job", data=data)

#Customer marital status distribution
sns.countplot(x="marital", data=data)

#Barplot for credit in default
sns.countplot(x="default", data=data)

#Barplot for housing loan
sns.countplot(x="housing", data=data)

#Barplot for personal loan
sns.countplot(x="loan", data=data)

#Barplot for previous marketing loan outcome
sns.countplot(x="poutcome", data=data)

#Dropping the redant columns
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                        18, 19]], axis=1, inplace=True)

#Creating Dummy Variables
data2 = pd.get_dummies(data, columns =['job', 'marital',
                                       'default', 'housing', 'loan', 'poutcome'])

#Drop the unknown columns
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
data2.columns

#Check the independence between the independent variables
sns.heatmap(data2.corr())
plt.show()

# Split the data into training and test sets
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# Fitting the Logistic Model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# Check for Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]

X_train.columns

New_X_train=X_train.drop(["job_admin."],axis = 1,inplace = True)
New_X_test=X_test.drop(["job_admin."],axis = 1,inplace = True)

New_X_train=X_train.drop(["marital_single"],axis = 1,inplace = True)
New_X_test=X_test.drop(["marital_single"],axis = 1,inplace = True)

# Fitting the Logistic Model Again 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

# Evaluating the Logistic Model
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix_train = confusion_matrix(y_train,y_pred_train)

print(confusion_matrix_train)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

##Computing false and true positive rates
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr,_=roc_curve(classifier.predict(X_test),y_test,drop_intermediate=False)

import matplotlib.pyplot as plt
##Adding the ROC
##Random FPR and TPR

##Title and label
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')

roc_auc_score(classifier.predict(X_test),y_test)


