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


def adaboost_svm(div_flag=False, myKernel='rbf', myDegree=1, myCoef0=1, early_stop=True,  debug=False):
    # boosted support vector machine (ensemble)
    svmb = AdaBoostSVM(C=150, gammaIni=5, myKernel=myKernel, myDegree=myDegree, myCoef0=myCoef0,
                       Diversity=div_flag, early_stop=early_stop, debug=debug)
    return svmb

def single_svm():
    # support vector machine (single case)
    return SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)

def linear_svm():
    # support vector machine (linear case)
    # decision
    return LinearSVC()

def bdt_svm():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(base_estimator=SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001),
                              n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=None)

def bag_svm():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier(base_estimator=SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001))

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

def model_loader_batch(process):
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    if(process==0):  return ("trad-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==1):  return ("trad-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==2):  return ("trad-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==3):  return ("trad-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)

    if(process==4):  return ("trad-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==5):  return ("trad-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==6):  return ("trad-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)
    if(process==7):  return ("trad-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0)

    if(process==8):  return ("genHLAUC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==9):  return ("genHLAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==10): return ("genHLAUC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==11): return ("genHLAUC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)

    if(process==12):  return ("genHLAUC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==13):  return ("genHLAUC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==14):  return ("genHLAUC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    if(process==15):  return ("genHLAUC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5)
    
    if(process==16):  return ("genHLACC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==17):  return ("genHLACC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==18):  return ("genHLACC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==19):  return ("genHLACC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)

    if(process==20):  return ("genHLACC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==21):  return ("genHLACC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==22):  return ("genHLACC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)
    if(process==23):  return ("genHLACC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5)

    if(process==24):  return ("genHLPREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==25):  return ("genHLPREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==26):  return ("genHLPREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==27):  return ("genHLPREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)

    if(process==28):  return ("genHLPREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==29):  return ("genHLPREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==30):  return ("genHLPREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)
    if(process==31):  return ("genHLPREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5)

    if(process==32):  return ("genHLF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==33):  return ("genHLF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==34):  return ("genHLF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==35):  return ("genHLF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)

    if(process==36):  return ("genHLF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==37):  return ("genHLF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==38):  return ("genHLF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    if(process==39):  return ("genHLF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5)
    
    if(process==40):  return ("genHLREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==41):  return ("genHLREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==42):  return ("genHLREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==43):  return ("genHLREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)

    if(process==44):  return ("genHLREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==45):  return ("genHLREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==46):  return ("genHLREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)
    if(process==47):  return ("genHLREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5)

    if(process==48):  return ("genHLGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==49):  return ("genHLGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==50):  return ("genHLGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==51):  return ("genHLGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)

    if(process==52):  return ("genHLGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==53):  return ("genHLGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==54):  return ("genHLGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    if(process==55):  return ("genHLGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5)
    
    # roulete
    if(process==56):  return ("genRLTAUC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==57):  return ("genRLTAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==58):  return ("genRLTAUC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==59):  return ("genRLTAUC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)

    if(process==60):  return ("genRLTAUC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==61):  return ("genRLTAUC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==62):  return ("genRLTAUC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    if(process==63):  return ("genRLTAUC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5)
    
    if(process==64):  return ("genRLTACC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==65):  return ("genRLTACC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==66):  return ("genRLTACC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==67):  return ("genRLTACC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)

    if(process==68):  return ("genRLTACC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==69):  return ("genRLTACC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==70):  return ("genRLTACC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)
    if(process==71):  return ("genRLTACC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5)

    if(process==72):  return ("genRLTPREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==73):  return ("genRLTPREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==74):  return ("genRLTPREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==75):  return ("genRLTPREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)

    if(process==76):  return ("genRLTPREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==77):  return ("genRLTPREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==78):  return ("genRLTPREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)
    if(process==79):  return ("genRLTPREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5)

    if(process==80):  return ("genRLTF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==81):  return ("genRLTF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==82):  return ("genRLTF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==83):  return ("genRLTF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)

    if(process==84):  return ("genRLTF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==85):  return ("genRLTF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==86):  return ("genRLTF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    if(process==87):  return ("genRLTF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5)
    
    if(process==88):  return ("genRLTREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==89):  return ("genRLTREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==90):  return ("genRLTREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==91):  return ("genRLTREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)

    if(process==92):  return ("genRLTREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==93):  return ("genRLTREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==94):  return ("genRLTREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)
    if(process==95):  return ("genRLTREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5)

    if(process==96):  return ("genRLTGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==97):  return ("genRLTGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==98):  return ("genRLTGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==99):  return ("genRLTGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)

    if(process==100):  return ("genRLTGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==101):  return ("genRLTGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==102):  return ("genRLTGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)
    if(process==103):  return ("genRLTGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5)

    

def model_loader(option=None, sample_name=None):
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)
    models_auc = []

    #if option == "b2b":
    # models_auc.append(("trad-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    # models_auc.append(("trad-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    # models_auc.append(("trad-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    # models_auc.append(("trad-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))

    # models_auc.append(("trad-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    models_auc.append(("trad-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    # models_auc.append(("trad-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))
    # models_auc.append(("trad-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "trad", 0.3, "auc", "roulette", 0.0))

    # models_auc.append(("genHLAUC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    # models_auc.append(("genHLAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    # models_auc.append(("genHLAUC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    # models_auc.append(("genHLAUC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))

    # models_auc.append(("genHLAUC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    # models_auc.append(("genHLAUC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    # models_auc.append(("genHLAUC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "highlow", 0.5))
    
    # models_auc.append(("genHLACC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    # models_auc.append(("genHLACC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    # models_auc.append(("genHLACC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    # models_auc.append(("genHLACC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))

    # models_auc.append(("genHLACC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    # models_auc.append(("genHLACC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))
    # models_auc.append(("genHLACC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "highlow", 0.5))

    # models_auc.append(("genHLPREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))

    # models_auc.append(("genHLPREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    models_auc.append(("genHLPREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))

    # models_auc.append(("genHLF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))

    # models_auc.append(("genHLF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    models_auc.append(("genHLF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    
    # models_auc.append(("genHLREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))

    # models_auc.append(("genHLREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    models_auc.append(("genHLREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))

    # models_auc.append(("genHLGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))

    # models_auc.append(("genHLGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    models_auc.append(("genHLGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    
    # roulete
    # models_auc.append(("genRLTAUC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    # models_auc.append(("genRLTAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    # models_auc.append(("genRLTAUC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    # models_auc.append(("genRLTAUC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))

    # models_auc.append(("genRLTAUC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    # models_auc.append(("genRLTAUC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    # models_auc.append(("genRLTAUC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "auc", "roulette", 0.5))
    
    # models_auc.append(("genRLTACC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    # models_auc.append(("genRLTACC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    # models_auc.append(("genRLTACC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    # models_auc.append(("genRLTACC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))

    # models_auc.append(("genRLTACC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    # models_auc.append(("genRLTACC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))
    # models_auc.append(("genRLTACC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "acc", "roulette", 0.5))

    # models_auc.append(("genRLTPREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    # models_auc.append(("genRLTPREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    # models_auc.append(("genRLTPREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    # models_auc.append(("genRLTPREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))

    # models_auc.append(("genRLTPREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    models_auc.append(("genRLTPREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    # models_auc.append(("genRLTPREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))
    # models_auc.append(("genRLTPREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "roulette", 0.5))

    # models_auc.append(("genRLTF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))

    # models_auc.append(("genRLTF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    models_auc.append(("genRLTF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    
    # models_auc.append(("genRLTREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))

    # models_auc.append(("genRLTREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    models_auc.append(("genRLTREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))

    # models_auc.append(("genRLTGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))

    # models_auc.append(("genRLTGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    models_auc.append(("genRLTGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=3, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))



    # if option == "boot" or option=="kfold":        
    #     models_auc.append((adaboost_svm(False), "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(True),  "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(False), "boost-svm", "absv",  "trad"))
    #     models_auc.append((adaboost_svm(True),  "boost-svm", "absv",  "trad"))
    # elif option == "no_div":                                
    #     models_auc.append((adaboost_svm(True),  "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(True),  "boost-svm", "absv",  "trad"))
    # elif option == "diverse":                               
    #     models_auc.append((adaboost_svm(False), "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(False), "boost-svm", "absv",  "trad"))

        
    # models_auc.append((single_svm(),  "single-svm", "prob", "single-svm", "trad"))
    # models_auc.append((linear_svm(),  "linear-svm", "deci", "linear-svm", "trad"))
    # models_auc.append((bdt_svm(),     "bdt-svm",    "prob", "bdt-svm",    "trad"))
    # models_auc.append((bag_svm(),     "bag-svm",    "prob", "bag-svm",    "trad"))
    # models_auc.append((rand_forest(), "rand-forest","prob", "rand-forest","trad"))
    # models_auc.append((bdt_forest(),  "bdt-forest", "prob", "bdt-forest", "trad"))
    # models_auc.append((bag_forest(),  "bag-forest", "prob", "bag-forest", "trad"))
    # models_auc.append((grad_forest(), "grad-forest","prob", "grad-forest","trad"))
    # models_auc.append((neural_net(),  "neural-net", "prob", "neural-net", "trad"))
    # models_auc.append((k_neighbors(), "k-neigh",    "prob", "k-neigh",    "trad"))
    # models_auc.append((gauss_nb(),    "gauss-nb",   "prob", "gauss-nb",   "trad"))
    # models_auc.append((gauss_pc(),    "gauss-pc",   "prob", "gauss-pc",   "trad"))
    # models_auc.append((log_reg(),     "log-reg",    "prob", "log-reg",    "trad"))
    # models_auc.append((ridge_class(), "ridge-cl",   "deci", "ridge-cl",   "trad"))
    # models_auc.append((sgdc_class(),  "sgdc-cl",    "deci", "sgdc-cl",    "trad"))
    # models_auc.append((pass_agre(),   "pass-agre",  "deci", "pass-agre",  "trad"))
    # if sample_name != "belle2_ii" and sample_name != "solar": # ugly fix
    #     models_auc.append((linear_dis(),  "prob", "linear-dis", "trad"))
    #     models_auc.append((quad_dis(),    "prob", "quad-dis", "trad"))

    return models_auc
