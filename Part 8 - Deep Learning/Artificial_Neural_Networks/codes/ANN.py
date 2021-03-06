# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:08:35 2017

@author: pshrivas
"""

#Artificial Neural Network

#Installing Theano
#pip install --upgrade --no-deps git+github.com/Theano/Theano.git
#Install TensorFlow

#install keras
#pip install --upgrade keras

# Part -1 Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
                
#Ecncoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()  
X = X[:,1:]            

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# ANN 

#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
#input node = no. of independent variable + output /2 in this case independent variables are 11 and output node is 1 so 11+1/2 = 6
classifier = Sequential()
#Adding input and hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#Adding Second hidden layer 
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#Adding Output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Stochastic Gradient Descent
#Compiling ANN

classifier.compile(optimizer="adam",loss = "binary_crossentropy", metrics = ["accuracy"])

#fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=20, epochs=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (1545+142)/2000