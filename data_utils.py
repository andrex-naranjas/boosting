#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM (data utils module)
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#metrics: some functions to measure the quality of the predictions
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

def cv_scores(model, x,y):
    scores=cross_val_score(model, x, y, cv=5)
    return "Cross-validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def generate_report(y_val, y_pred):
    print('Accuracy = ', round(accuracy_score(y_val, y_pred) * 100, 2))
    print('Precision = ', round(precision_score(y_val, y_pred) * 100 ,2))
    print('Recall = ', round(recall_score(y_val, y_pred) * 100, 2))
    print('f1_score =', round(f1_score(y_val, y_pred) * 100, 2))
    pass

def generate_auc_roc_curve(model, X_val,Y_test, name):
    Y_pred_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    auc = round(roc_auc_score(Y_test, Y_pred_prob) *100 ,2)
    string_model= str(model)
    plt.plot(fpr, tpr, label = 'AUC ROC ' + string_model[:3] + '=' + str(auc))
    plt.legend(loc = 4)
    plt.savefig(name+'.pdf')
    #pass
    return

def metrics(name, method, X_train, Y_train, Y_test, X_test, Y_pred):
    print('\n '+name+': ')
    print(cv_scores(method, X_train, Y_train))
    generate_report(Y_test, Y_pred)
    generate_auc_roc_curve(method, X_test,Y_test,name)
    return
