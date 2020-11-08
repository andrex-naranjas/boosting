#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------
Code to improve Adaptive Boosted Support Vector Machines
Authors: A. Ramirez-Morales and J. Salmon-Gamboa

Model comparison module
--------------------------------------------------------
'''

# machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# import boostedSVM class
from boostedSVM import AdaBoostSVM, Div_AdaBoostSVM

# import module for data utils
import data_utils as du
import data_visualization as dv

import pandas as pd
from datetime import datetime as dt

# fit and predict model
def fit_predict(sample, model_name, model, X_train, Y_train, X_test, Y_test):

    model.fit(X_train, Y_train)
    start =  dt.now()
    y_pred = model.predict(X_test)
    end = dt.now()
    time = end-start

    # performs ROC calculations for AdaBoostSVMs
    if model_name == 'AB-SVM' or model_name == 'DivAB-SVM':
        print('Analysing sample: ', sample)
        y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
        TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)

        # creates ROC files
        dv.plot_roc_curve(model_name,TPR,FPR,sample,'real',   glob_local=True)
        dv.plot_roc_curve(model_name,TPR,FPR,sample,'sorted', glob_local=True)
        print(f'End of{model_name}')

    return time, y_pred

# perform the ml algos and generate report (fit , predict and metrics)
def comparison(sample, X_train, Y_train, X_test, Y_test):

    # initialise list to store process times
    times_list = ['Time']

    # creating the dataframe containing the output metrics
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_df = pd.DataFrame({'Metric': metrics_list})

    classifiers = {
        'rForest' : RandomForestClassifier(n_estimators=100),
        'AdaBoost-SAMME' : AdaBoostClassifier(n_estimators=100, random_state=0),
        'NeuralNet' : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        'XGBoost' : GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'SVC' : SVC(gamma='auto', probability = True),
        'KNN' : KNeighborsClassifier(n_neighbors=3),
        'AB-SVM' : AdaBoostSVM(C=50, gammaIni=100),
        'DivAB-SVM' : Div_AdaBoostSVM(C=50, gammaIni=100)
        }

    for model in zip(classifiers.keys(), classifiers.values()):
        time, Y_pred = fit_predict(sample, model[0],model[1], X_train, Y_train, X_test, Y_test)
        times_list.append(time)
        temp_list = du.metrics(sample,model[0],model[1], X_train, Y_train, Y_test, X_test, Y_pred)
        temp_df = pd.DataFrame({model[0]: temp_list})
        metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # save csv
    metrics_df.loc[6] = times_list
    metrics_df.to_csv('output/{}/metrics_report.csv'.format(sample), index=False)
