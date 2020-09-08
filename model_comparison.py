#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to improve SVM (machine learning methods module)
# authors: A. Ramirez-Morales and J. Salmon-Gamboa

# model comparison module

# machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# import module for data utils
import data_utils as du
import pandas as pd


# perform the ml algos and generate report (fit , predict and metrics)
def comparison(sample, X_train, Y_train, Y_test, X_test):

    # creating the dataframe containing the output metrics
    metrics_list = ['Cross Validation', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_df = pd.DataFrame({'Metric': metrics_list})

    # RANDOM forest classifier
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    temp_list = du.metrics(sample, 'rForest',random_forest, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'Random Forest': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
    AdaBoost.fit(X_train, Y_train)
    Y_pred = AdaBoost.predict(X_test)
    temp_list = du.metrics(sample, 'AdaBoost-SAMME', AdaBoost, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'AdaBoost': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # Neural Network Multi Layer Perceptron classifier
    NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    NeuralNet.fit(X_train, Y_train)
    Y_pred = NeuralNet.predict(X_test)
    temp_list = du.metrics(sample, 'NeuralNet', NeuralNet, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'NeuralNet': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # Gradient Boost Classifier XGBoost
    model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model_GBC.fit(X_train, Y_train)
    Y_pred=model_GBC.predict(X_test)
    temp_list = du.metrics(sample, 'XGBoost', model_GBC, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'XGBoost': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # support vector machine
    svc = SVC(gamma='auto', probability = True)
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    temp_list = du.metrics(sample, 'SVC', svc, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'SVC': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    metrics_df.to_csv('output/{}/metrics_report.csv'.format(sample), index=False)
