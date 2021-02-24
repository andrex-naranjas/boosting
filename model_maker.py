#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# model caller module

# support vector machines
from boostedSVM import AdaBoostSVM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# n-word classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# probabilistic
from sklearn.naive_bayes      import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

# linear models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

# import basics
import numpy as np


def adaboost_svm(div_flag=False, myKernel='rbf'):
    return AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel, Diversity=div_flag)    

def single_svm():
    # support vector machine (single case)
    return SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)

def linear_svm():
    # support vector machine (linear case)
    # decision
    return LinearSVC()

def bdt_svm():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(base_estimator=SVC(), n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=None)

def bag_svm():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier(base_estimator=SVC())

def rand_forest():
    # RANDOM forest classifier
    return RandomForestClassifier(n_estimators=100)

def bdt_forest():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(n_estimators=100, random_state=None)

def bag_forest():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier()

def grad_forest():
    # gradient boost classifier tree, this only for trees!
    return GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=None)
    
def neural_net():
    # Neural Network Multi Layer Perceptron classifier
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)

def k_neighbors():
    # K neighbors classifier. n_neighbors=3 because there are 2 classes
    return KNeighborsClassifier(n_neighbors=3)

def linear_dis():
    # to-do set values
    return LinearDiscriminantAnalysis()

def quad_dis():
    # to-do set values
    return QuadraticDiscriminantAnalysis()

def gauss_nb():
    # to-do: set values    
    return GaussianNB()

def gauss_pc():
    # to-do: set values
    return GaussianProcessClassifier()

def log_reg():
    # to-do: set values
    return LogisticRegression()

def ridge_class():
    # to-do: set values
    # decision
    return RidgeClassifier()

def sgdc_class():
    # to-do: set values
    # decision
    return SGDClassifier()

def pass_agre():
    # to-do: set values
    # decision
    return PassiveAggressiveClassifier()
