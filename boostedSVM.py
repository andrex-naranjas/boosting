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

# perform the ml algos
# support vector machine
svc = SVC(gamma='auto', probability = True)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
du.metrics('svm', svc, X_train, Y_train, Y_test, X_test, Y_pred)

# comparison with other models (fit, predict and metrics)
mc.comparison(X_train, Y_train, Y_test, X_test, Y_pred)
