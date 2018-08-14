# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:37:13 2017

@author: pshrivas
"""


# Importing Libraries
# numpy includes mathematical tools
# matplotlib plot charts
# pandas import and manage datasets
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Splitting dataset into training set and test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#==============================================================================
# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
#  #for training we need to fit and then transform for test we have to only transform.
# X_test = sc_X.transform(X_test)
#==============================================================================

#Fitting Random Forest Regression Model to the dataset    
#Create a regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor.fit(X,y)

#Predicting a new result with Regression results
y_pred = regressor.predict(6.5)



#Visualising the Linear regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 

plt.scatter(X,y,color = "red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluff(Random forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()