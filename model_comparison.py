#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to improve SVM (machine learning methods module)
# authors: A. Ramirez-Morales and J. Salmon-Gamboa

# machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# import module for data utils
import data_utils as du

# perform the ml algos (fit , predict and metrics)
def comparison(X_train, Y_train, Y_test, X_test, Y_pred):
    
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

    # Neural Network Multi Layer Perceptron classifier
    NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    NeuralNet.fit(X_train, Y_train)
    Y_pred = NeuralNet.predict(X_test)
    du.metrics('NeuralNet',NeuralNet, X_train, Y_train, Y_test, X_test, Y_pred)

    # Gradient Boost Classifier XGBoost
    model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model_GBC.fit(X_train, Y_train)
    Y_pred=model_GBC.predict(X_test)
    du.metrics('XGBoost',model_GBC, X_train, Y_train, Y_test, X_test, Y_pred)