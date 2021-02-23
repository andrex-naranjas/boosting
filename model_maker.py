#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# model maker module

# machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from boostedSVM import AdaBoostSVM

# import module for data utils
import data_utils as du
import pandas as pd
import datetime
import numpy as np


def adaboost_svm(div_flag=False, myKernel='rbf'):
    return AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel, Diversity=div_flag)    

def single_svm():
    # support vector machine (single case)
    return SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)

def rand_forest():
    # RANDOM forest classifier
    return RandomForestClassifier(n_estimators=100)

def bdt_forest():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(n_estimators=100, random_state=0)
    
def neural_net():
    # Neural Network Multi Layer Perceptron classifier
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)

def k_neighbors():
    # K neighbors classifier. n_neighbors=3 because there are 2 classes
    return KNeighborsClassifier(n_neighbors=3)

def xgboost_tree():
    # Gradient Boost Classifier XGBoost
    return GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)
        
