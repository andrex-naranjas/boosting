#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code to improve SVM
# authors: A. Ramirez-Morales and J. Salmon-Gamboa

# main module

# python basics
import sys
import numpy as np
import pandas as pd

# machine learning
from sklearn.svm import SVC, LinearSVC

# class for data preparation
from utils import data_preparation.data_preparation as data_preparation

# module for data utils
import data_utils as du

# ML module for comparison
import model_comparison as mc

# AdaBoost class
from boostedSVM import AdaBoostSVM

# data visualization module
from utils import data_visualization as dv


# start of the module
# make directories
sample_list = ['titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe']
du.make_directories(sample_list)

# get the data
#'titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe'
data = data_preparation()
sample = 'titanic' # heart (issues); two_norm, nursery(large)
X_train, Y_train, X_test, Y_test = data.dataset(sample,'',False,0.4)

# run single support vector machine
weights= np.ones(len(Y_train))/len(Y_train)
svc = SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)
svc.fit(X_train, Y_train, weights)
Y_pred = svc.predict(X_test)
du.metrics(sample,'svm', svc, X_train, Y_train, Y_test, X_test, Y_pred)

# comparison with other ml models (fit, predict and metrics)
#mc.comparison(sample, X_train, Y_train, Y_test, X_test)
#du.cv_metrics(model, X_train, Y_train)

# run AdaBoost support vector machine
model = AdaBoostSVM(C = 50, gammaIni = 10)
model.fit(X_train, Y_train)
y_preda = model.predict(X_test)

# check model performance
test_pre = (model.predict(X_test) == Y_test).mean()
test_err = (model.predict(X_test) != Y_test).mean()
print(f'Test prec.: {test_pre:.1%}')
print(f'Test error: {test_err:.1%}')


# metrics for plots
weights, errors, precision = model.get_metrics()

# precision plot
dv.plot_frame(pd.DataFrame(precision*100,np.arange(precision.shape[0])),
                           'Classifier precision', 'Classifier', 'training precision (%)', True, 0, 100,'titanic')
# errors plot
dv.plot_frame(pd.DataFrame(errors*100,np.arange(errors.shape[0])),
                           'Classifier error', 'Classifier', 'training error (%)', True, 0, 100,'titanic')

# weights plot
dv.plot_frame(pd.DataFrame(weights[10],np.arange(weights.shape[1])),
                           'Sample weights', 'Sample', 'weights (a.u.)', True, -0.005, 0.01,'titanic')

# grid hyper parameter 2D-plots
matrix = du.grid_param_gauss(X_train, Y_train, X_test, Y_test, sigmin=-5, sigmax=5, cmin=0, cmax=6)
dv.plot_2dmap(matrix,-5,5,0,6,'titanic')

# boostrap error VS number of classiffiers calculation
frame = du.error_number('titanic',myC=50,myGammaIni=10)
dv.plot_frame(frame, 'Classifiers error', 'No. Classifiers', 'test error', False, 0, 50,'titanic')
