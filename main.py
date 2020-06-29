#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code to improve SVM
# authors: A. Ramirez-Morales and J. Salmon-Gamboa

# main module

# python basics
import sys

# data analysis and wrangling
import numpy as np
import pandas as pd

# machine learning
from sklearn.svm import SVC, LinearSVC

# import class for data preparation
from data_preparation import data_preparation

# import module for data utils
import data_utils as du

# import ML models for comparison
import model_comparison as mc

# import AdaBoost
from boostedSVM import AdaBoostSVM

# start of the module

# make directories
sample_list = ['titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe']
du.make_directories(sample_list)

# run the calculations
# get the data
#'titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe'
data = data_preparation()
sample = 'titanic' # heart (issues); two_norm, nursery(large)
X_train, Y_train, X_test, Y_test = data.dataset(sample,'',False,0.4)

# single support vector machine
weights= np.ones(len(Y_train))/len(Y_train)
svc = SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)
svc.fit(X_train, Y_train, weights)
Y_pred = svc.predict(X_test)
du.metrics(sample,'svm', svc, X_train, Y_train, Y_test, X_test, Y_pred)

# comparison with other ml models (fit, predict and metrics)
#mc.comparison(sample, X_train, Y_train, Y_test, X_test)
#du.cv_metrics(model, X_train, Y_train)

#AdaBoost support vector machine
model = AdaBoostSVM(C = 150, gammaIni = 10)
model.fit(X_train, Y_train)
y_preda = model.predict(X_test)

# check model performance
test_pre = (model.predict(X_test) == Y_test).mean()
test_err = (model.predict(X_test) != Y_test).mean()
print(f'Test prec.: {test_pre:.1%}')
print(f'Test error: {test_err:.1%}')

# boostrap error VS number of classiffiers calculation
frame = du.error_number('titanic',150,10)

import matplotlib.pyplot as plt
frame.plot()
plt.show()
    

