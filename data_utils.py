#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM (data utils module)
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#metrics: some functions to measure the quality of the predictions
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold

import numpy as np
import math as math

# import class for data preparation
from data_preparation import data_preparation

# import boostedSVM class
from boostedSVM import AdaBoostSVM

# machine learning
from sklearn.svm import SVC, LinearSVC

# bootstrap
from sklearn.utils import resample

# makes a directory for each dataset
def make_directories(sample_list):
    for item in sample_list:
        try:
            os.makedirs('output/{}'.format(item))
        except FileExistsError:
            pass

def cv_scores(model, x,y):
    scores = cross_val_score(model, x, y, cv=5)
    print("Cross-validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return ["%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)]


# Makeshift metric for predictors
def cv_metrics(model, X, y):
    
    X = X.values
    y = y.values
    
    kf = KFold(n_splits = 5, shuffle = True)
    
    acc_scores = np.array([])
    prec_scores = np.array([])
    recall_scores = np.array([])
    f1_scores = np.array([])
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_predicted)
        prec = precision_score(y_test, y_predicted)
        recall = recall_score(y_test, y_predicted)
        f1 = f1_score(y_test, y_predicted)
        
        acc_scores = np.append(acc_scores, acc)
        prec_scores = np.append(prec_scores, prec)
        recall_scores = np.append(recall_scores, recall)
        f1_scores = np.append(f1_scores, f1)
        
    print("Cross-validation Accuracy Score: %0.2f (+/- %0.2f)" % (acc_scores.mean(), acc_scores.std() * 2))
    print("Cross-validation Precision Score: %0.2f (+/- %0.2f)" % (prec_scores.mean(), prec_scores.std() * 2))
    print("Cross-validation Recall Score: %0.2f (+/- %0.2f)" % (recall_scores.mean(), recall_scores.std() * 2))
    print("Cross-validation F1 Score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))


def generate_report(y_val, y_pred, verbose):
    acc = round(accuracy_score(y_val, y_pred) * 100, 2)
    prec = round(precision_score(y_val, y_pred) * 100 ,2)
    recall = round(recall_score(y_val, y_pred) * 100, 2)
    f1 =  round(f1_score(y_val, y_pred) * 100, 2)

    if verbose:
        print('Accuracy = ', acc)
        print('Precision = ', prec)
        print('Recall = ', recall)
        print('f1_score =', f1)

    return [acc, prec, recall, f1]

def generate_auc_roc_curve(sample, model, X_val, Y_test, name):
    Y_pred_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    auc = round(roc_auc_score(Y_test, Y_pred_prob) *100 ,2)
    string_model= str(model)
    #plt.plot(fpr, tpr, label = 'AUC ROC ' + string_model[:3] + '=' + str(auc))
    #plt.legend(loc = 4)
    #plt.savefig(name+'.pdf')
    output = pd.DataFrame({'False positive rate': fpr,'True positive rate': tpr})
    output.to_csv('output/' + sample +  '/' + string_model[:3] + 'roc.csv', index=False)
    return

def metrics(sample, name, method, X_train, Y_train, Y_test, X_test, Y_pred):
    generate_auc_roc_curve(sample, method, X_test,Y_test, name)
    print('\n '+name+': ')
    return cv_scores(method, X_train, Y_train) + generate_report(Y_test, Y_pred, verbose=True)


# function to get average errors via bootstrap, for 1-n classifiers
def error_number(sample_name, myC, myGammaIni):

    # fetch data_frame without preparation
    data_df   = data_preparation()
    sample_df = data_df.fetch_data(sample_name)

    #prepare bootstrap sample
    total = []
    number = ([])
    
    for _ in range(100): # arbitrary number of samples to produce
        sampled_data = resample(sample_df, replace = True, n_samples = 400, random_state = 0)
        data = data_preparation()

        X_train, Y_train, X_test, Y_test = data.dataset(sample_name,sampled_data,True,0.4)
        
        # run AdaBoostSVM (train the model)
        model = AdaBoostSVM(C = myC, gammaIni = myGammaIni)
        model.fit(X_train, Y_train)
        
        # compute test samples
        test_number = model.number_class(X_test)
        number = np.append(number, [len(test_number)])
        error = ([])
        for i in range(len(test_number)):
            error_d = 0
            error_d = (test_number[i] != Y_test).mean()
            error   = np.append(error, [round(error_d * 100, 2)])
            
            total.append(error)
            
    total = np.array(total)

    # complete total with nan's for dimension consistency
    total_final = []
    for i in range(len(total)):    
        if(len(total[i]) < np.amax(number)):
            for _ in range(int(np.amax(number) - len(total[i]))):
                total[i] = np.append(total[i], [np.nan])
                
        total_final.append(total[i])
                                                
    total_final = np.array(total_final)                        
    final_final = np.nanmean(total_final,axis=0)
    
    return pd.DataFrame(final_final,np.arange(np.amax(number)))


# grid svm-hyperparameters (sigma and C) to explore test errors
def grid_param_gauss(train_x, train_y, test_x, test_y, sigmin, sigmax, cmin, cmax):

    # inverted limits, to acommodate the manner at which the arrays are stored and plotted as a matrix
    log_step_c     = np.logspace(cmax,cmin,15,endpoint=True,base=math.e)
    log_step_sigma = np.logspace(sigmax,sigmin,15,endpoint=True,base=math.e)
    
    error_matrix = []
    for i in range(len(log_step_c)): # C loop
        errors = ([])
        for j in range(len(log_step_sigma)): # sigma loop
            svc = SVC(C= log_step_c[i], kernel='rbf', gamma=1/(2*((log_step_sigma[j])**2)), shrinking = True, probability = True, tol = 0.001)
            svc.fit(train_x, train_y)
            pred_y = svc.predict(test_x)
            acc, prec, recall, f1 = generate_report(test_y, pred_y, verbose=False)
            errors = np.append(errors,[(0.01)*(100-acc)])
            
        error_matrix.append(errors)
        
    return np.array(error_matrix)
