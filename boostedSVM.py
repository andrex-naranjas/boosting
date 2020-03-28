#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

#Main module

# python basics
import sys

# data analysis and wrangling
import numpy as np
import random as rnd

# machine learning
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split

# import module for data preparation
import data_preparation as dp

# import module for data utils
import data_utils as du

# import ML models for comparison
import model_comparison as mc

if len(sys.argv) != 2:
    sys.exit('Please provide which sample name to analyse. Try again!') 

sample = sys.argv[1]

# fetch data set (from available list)
data_set = dp.fetch_data(sample)

# check data
print("Before preparation", data_set.shape)
print(data_set.columns.values)
print(data_set.head())
print(data_set.tail())
print(data_set.describe())

# prepare data
if sample == 'titanic':
    X,Y = dp.titanic(data_set)
elif sample == 'two_norm':
    X,Y = dp.two_norm(data_set)
elif sample == 'cancer':
    X,Y = dp.bCancer(data_set)
elif sample == 'german':
    X,Y = dp.german(data_set)
elif sample == 'heart':
    X,Y = dp.heart(data_set)
    
# print data after preparation
print("After preparation", data_set.shape)
print(X.head())

# divide sample into train and test sample
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# support vector machine
# nominal 
svc = SVC(gamma='auto', probability = True)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
du.metrics('svm', svc, X_train, Y_train, Y_test, X_test, Y_pred)

# boosting
# initialize 
gammaIni,gammaMin,gammaStep,gammaVar = 7.01,0.07,0.1,0.0
cost,count = 1, 0
y = []
weights = []
h_list = []
alpha_list = []

for var in Y_train:
     y.append(var)
     weights.append(1.0/len(Y_train))
     
# svm function, keeps the error below 50%
def svc_train(myKernel, myGamma, iniGamma, stepGamma, y, x_train, y_train, myWeights, count):

    if count == 0:
        myGamma = iniGamma
        
    errorOut = 0.0
    hOut = []
    while True:
        svcB = SVC(C=1.0, kernel='rbf', gamma=myGamma, shrinking = True, probability = True, tol = 0.001)
        svcB.fit(x_train, y_train)
        y_pred = svcB.predict(x_train)
        
        hOut = []
        for var in y_pred:
            hOut.append(var)
        
        for i in range(len(y_pred)):
            if(y[i]!=hOut[i]):
                errorOut+=myWeights[i]

        if errorOut < 0.5:
            myGamma -= stepGamma
            break
        
        myGamma -= stepGamma
                    
    return myGamma,errorOut,hOut

# AdaBoost loop
while True:

    gammaVar,error,h = svc_train('rbf', gammaVar, gammaIni, gammaStep, y, X_train, Y_train, weights, count)

    h_list.append(h) 
        
    x = (1 - error)/error
    alpha = 0.5 * np.log(x)
    alpha_list.append(alpha)
    new_weights = []

    for i in range(len(y)):
        x = (-1.0) * alpha * y[i] * h[i]
        new_weights = weights[i] * np.exp(x)
        
    count+=1
    print(gammaVar,' :gamma')
        
    if gammaVar < gammaMin:
        break

#du.metrics('svmBoosted', svcB, X_train, Y_train, Y_test, X_test, Y_predB)


# comparison with other ml models (fit, predict and metrics)
#mc.comparison(X_train, Y_train, Y_test, X_test)
