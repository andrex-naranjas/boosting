# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:09:24 2020

@author: tokam
"""

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

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

# import module for data preparation
import data_preparation as dp

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

# Metrics - Some functions to measure the quality of the predictions
def cv_scores(model, x,y):
    scores=cross_val_score(model, x, y, cv=5)
    return "Cross-validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def generate_report(y_val, y_pred):
    print('Accuracy = ', round(accuracy_score(y_val, y_pred) * 100, 2))
    print('Precision = ', round(precision_score(y_val, y_pred) * 100 ,2))
    print('Recall = ', round(recall_score(y_val, y_pred) * 100, 2))
    print('f1_score =', round(f1_score(y_val, y_pred) * 100, 2))
    pass

def generate_auc_roc_curve(model, X_val):
    Y_pred_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    auc = round(roc_auc_score(Y_test, Y_pred_prob) *100 ,2)
    string_model= str(model)
    plt.plot(fpr, tpr, label = 'AUC ROC ' + string_model[:3] + '=' + str(auc))
    plt.legend(loc = 4)
    plt.show()
    pass

# perform the ml algos
# support vector machine
svc = SVC(gamma='auto', probability = True)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print('\n SVM: ')
print(cv_scores(svc, X_train, Y_train))
generate_report(Y_test, Y_pred)
generate_auc_roc_curve(svc, X_test)


# random forest classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print('\n Random Forest: ')
print(cv_scores(random_forest, X_train, Y_train))
generate_report(Y_test, Y_pred)
generate_auc_roc_curve(random_forest, X_test)

# AdaBoost This class implements the algorithm known as AdaBoost-SAMME:
# Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
AdaBoost.fit(X_train, Y_train)
Y_pred = AdaBoost.predict(X_test)
print('\n AdaBoost-SAMME: ')
print(cv_scores(AdaBoost, X_train, Y_train))
generate_report(Y_test, Y_pred)
generate_auc_roc_curve(AdaBoost, X_test)

# Neural Network Multi Layer perceptron classifier
NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NeuralNet.fit(X_train, Y_train)
Y_pred = NeuralNet.predict(X_test)
print('\n Neural Networks MLPC: ')
print(cv_scores(NeuralNet, X_train, Y_train))
generate_report(Y_test, Y_pred)
generate_auc_roc_curve(NeuralNet, X_test)

# Gradient Boost Classifier XGBoost
model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model_GBC.fit(X_train, Y_train)
Y_pred=model_GBC.predict(X_test)
print('\n XGBoost: ')
print(cv_scores(model_GBC, X_train, Y_train))
generate_report(Y_test, Y_pred)
generate_auc_roc_curve(model_GBC, X_test)