#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

#Main module

# python basics
import sys

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# machine learning
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# import module for data preparation
import data_preparation as dp

# import module for data utils
import data_utils as du


if len(sys.argv) != 2:
    sys.exit('Please provide which sample name to analyse. Try again!') 

sample = sys.argv[1]

# fetch data
if sample == 'titanic':
    data_set = pd.read_csv('./data/titanic.csv')
elif sample == 'two_norm':
    data_set = pd.read_csv('./data/two_norm.csv')
elif sample == 'cancer':
    data_set = pd.read_csv('./data/breast_cancer.csv')
elif sample == 'german':
    data_set = pd.read_csv('./data/german.csv')
else:
    sys.exit('The sample name provided does not exist. Try again!')

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

# RANDOM forest classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
du.metrics('rForest',random_forest, X_train, Y_train, Y_test, X_test, Y_pred)

# AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
AdaBoost.fit(X_train, Y_train)
Y_pred = AdaBoost.predict(X_test)
du.metrics('AdaBoost-SAMME',AdaBoost, X_train, Y_train, Y_test, X_test, Y_pred)

# Neural Network Multi Layer perceptron classifier
NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NeuralNet.fit(X_train, Y_train)
Y_pred = NeuralNet.predict(X_test)
du.metrics('NeuralNet',NeuralNet, X_train, Y_train, Y_test, X_test, Y_pred)

# Gradient Boost Classifier XGBoost
model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model_GBC.fit(X_train, Y_train)
Y_pred=model_GBC.predict(X_test)
du.metrics('XGBoost',model_GBC, X_train, Y_train, Y_test, X_test, Y_pred)
