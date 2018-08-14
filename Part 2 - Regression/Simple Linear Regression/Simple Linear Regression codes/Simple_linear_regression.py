# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:18:54 2017

@author: pshrivas
"""

#==============================================================================
#Simple Linear Regression 
#==============================================================================
# Importing Libraries
# numpy includes mathematical tools
# matplotlib plot charts
# pandas import and manage datasets
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing Dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#==============================================================================
# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
#  #for training we need to fit and then transform for test we have to only transform.
# X_test = sc_X.transform(X_test)
#==============================================================================
#==============================================================================
#Fitting Simple Linear Regression to the Training set 
#==============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#==============================================================================
# Predicting the Test results
#==============================================================================
y_pred = regressor.predict(X_test)

#==============================================================================
# Visualising the Training set result
#==============================================================================
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('$Salary')
plt.show()

#==============================================================================
# Visualising the Test set result
#==============================================================================
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('$Salary')
plt.show()



























