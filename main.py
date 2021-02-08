#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# main module
import sys
import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVC, LinearSVC # machine learning

# framework includes
from data_preparation import data_preparation
import data_utils as du
import model_comparison as mc
import data_visualization as dv
from boostedSVM import AdaBoostSVM
from model_performance import model_performance
from sklearn.metrics import accuracy_score


# make directories
sample_list = ['titanic', 'cancer', 'german', 'heart', 'solar','car','contra','tac_toe', 'belle2_i', 'belle2_ii','belle_iii']
du.make_directories(sample_list)

# kernel selection
kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
myKernel = 'rbf'

# get the data
data = data_preparation()
sample_list = ['belle2_ii']

# loop over datasets in sample_list for AdaBoostSVM and other classifiers. get ROC curves & metrics
for name in sample_list:
    print('Analysing sample: ', name)
    split_flag = False
    if name=='belle2_iii': split_flag = True

    X_train, Y_train, X_test, Y_test = \
    data.dataset(sample_name=name,
                 sampling=False,split_sample=0.4,train_test=split_flag)

    # run AdaBoost support vector machine
    print('AdaBoost-support vector machines')
    model = AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel)

    start = datetime.datetime.now()
    model.fit(X_train, Y_train)
    end = datetime.datetime.now()
    elapsed_time = pd.DataFrame({'Elapsed time': [end - start]})

    elapsed_time.to_csv('output/' + name +  '/' + 'AdaBoostSVM_time.csv', index=False)
    y_preda = model.predict(X_test)
    print('Final test prediction:   ', accuracy_score(Y_test, y_preda))
    y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
    TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)

    nWeaks = len(model.alphas) # print on plot no. classifiers
    # dv.plot_roc_curve(TPR,FPR,sample,'real',   glob_local=True, name='nom')
    dv.plot_roc_curve(TPR,FPR,name,'sorted', glob_local=True, name='nom', kernel=myKernel, nClass=nWeaks)
    print('End adaboost')

    # run Diverse-AdaBoost Diversity support vector machine
    print('Diverse-AdaBoost-support vector machines')
    model_a = AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel, Diversity=True)
    model_a.fit(X_train, Y_train)
    nWeaks=len(model_a.alphas)
    y_preda_a = model_a.predict(X_test)
    print('Final test prediction:   ', accuracy_score(Y_test, y_preda_a))
    y_thresholds_a = model_a.decision_thresholds(X_test, glob_dec=True)
    TPR_a, FPR_a = du.roc_curve_adaboost(y_thresholds_a, Y_test)

    # print(model_a.diversities)

    # dv.plot_roc_curve(TPR_a,FPR_a,sample,'real',   glob_local=True, name='div')
    dv.plot_roc_curve(TPR_a,FPR_a,name,'sorted', glob_local=True, name='div', kernel=myKernel, nClass=nWeaks)
    print('End adaboost')
    
    # comparison with other ml models (fit, predict and metrics)
    mc.comparison(name, X_train, Y_train, Y_test, X_test)
    # metrics (via cross-validation)
    # du.cv_metrics(model, X_train, Y_train)

performance = model_performance(model, X_train, Y_train, X_test, Y_test)
