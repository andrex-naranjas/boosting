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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

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

def generate_report(y_val, y_pred):
    acc = round(accuracy_score(y_val, y_pred) * 100, 2)
    prec = round(precision_score(y_val, y_pred) * 100 ,2)
    recall = round(recall_score(y_val, y_pred) * 100, 2)
    f1 =  round(f1_score(y_val, y_pred) * 100, 2)

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
    return cv_scores(method, X_train, Y_train) + generate_report(Y_test, Y_pred)
