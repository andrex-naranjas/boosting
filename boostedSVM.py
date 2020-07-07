#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

#Main module

# python basics
import sys

# data analysis and wrangling
import numpy as np
import random as rnd
import pandas as pd

# machine learning
from sklearn.svm import SVC, LinearSVC

# import class for data preparation
from data_preparation import data_preparation

# import module for data utils
import data_utils as du

# import ML models for comparison
import model_comparison as mc

# make directories
sample_list = ['titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe']
du.make_directories(sample_list)

# main function

class AdaBoostSVM:

    def __init__(self, C, gammaIni):

        self.C = C
        self.gammaIni = gammaIni
        self.weak_svm = ([])
        self.alphas = ([])


    def _check_X_y(self, X, y):

        # Validate assumptions about format of input data. Expecting response variable to be formatted as ±1
        assert set(y) == {-1, 1}

        # If input data already is numpy array, do nothing
        if type(X) == type(np.array([])) and type(y) == type(np.array([])):
            return X, y
        else:
            # convert pandas into numpy arrays
            X = X.values
            y = y.values
            return X, y


    def svc_train(self, myKernel, myGamma, stepGamma, x_train, y_train, myWeights):

        if self.count == 0:
            myGamma = self.gammaIni

        while True:
            if myGamma<0:
                break

            errorOut = 0.0

            svcB = SVC(C = self.C, kernel='rbf', gamma=1/(2*(myGamma**2)), shrinking = True, probability = True, tol = 0.001)
            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)

            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            # require an error below 50% and avoid null errors
            if errorOut < 0.5 and errorOut != 0:
                myGamma -= stepGamma
                break

            myGamma -= stepGamma

        return myGamma, errorOut, y_pred, svcB


    def fit(self, X, y):

        X_train, Y_train = self._check_X_y(X, y)
        n = X.shape[0]
        weights= np.ones(n)/n

        gammaMin, gammaStep, gammaVar = 0.1, 0.1, 0.0
        cost, self.count, norm = 1, 0, 0.0
        h_list = []

        # AdaBoost loop
        while True:
            if self.count == 0:
                norm = 1.0
                new_weights = weights.copy()

            new_weights = new_weights/norm

            # call svm, weight samples, iterate sigma(gamma), get errors, obtain predicted classifier (h as an array)
            gammaVar, error, h, learner = self.svc_train('rbf', gammaVar, gammaStep, X_train, Y_train, new_weights)

            # count how many times SVM runs
            self.count += 1

            # calculate precision
            fp,tp = 0,0
            for i in range(n):
                if(Y_train[i]!=h[i]):
                    fp+=1
                else:
                    tp+=1

            # store the predicted classes
            h_temp = h.tolist()
            h_list.append(h_temp)

            #print("Error: {} Precision: {} Gamma: {} ".format(round(error,4), round(tp / (tp + fp),4), round(gammaVar+gammaStep,2)))
            # classifier weights (alpha), obtain and store
            x = (1 - error)/error
            alpha = 0.5 * np.log(x)
            self.alphas = np.append(self.alphas, alpha)
            self.weak_svm = np.append(self.weak_svm, learner)

            # reset weight lists
            weights = new_weights.copy()
            new_weights = ([])
            norm = 0.0

            # set weights for next iteration
            for i in range(n):
                x = (-1.0) * alpha * Y_train[i] * h[i]
                new_weights = np.append(new_weights, [weights[i] * np.exp(x)] )
                norm += weights[i] * np.exp(x)

            # do loop as long gamma > gammaMin, if gamma < 0, SVM fails exit loop
            if gammaVar < gammaMin:#) or (gammaVar < 0):
                break


        # h_list into array
        h_list = np.array(h_list)

        print(self.count,'number of classifiers')

        # start to calculate the final classifier
        h_alpha = np.array([h_list[i]*self.alphas[i] for i in range(self.count)])

        final = ([]) # final classifier is an array (size of number of data points)
        for j in range(len(h_alpha[0])):
            suma = 0.0
            for i in range(self.count):
                suma+=h_alpha[i][j]
            final = np.append(final, [np.sign(suma)])

        # final precision calculation
        final_fp, final_tp  = 0, 0
        for i in range(n):
            if(Y_train[i]!=final[i]):
                final_fp+=1
            else:
                final_tp+=1

        final_precision = final_tp / (final_fp + final_tp)
        print("Final Precision: {} ".format( round(final_precision,4)) )
        #du.metrics(sample,'svmBoosted', svcB, X_train, Y_train, Y_test, X_test, Y_predB)

        return self


    def boost_predict(self, X):
        # Make predictions using already fitted model
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        return np.sign(np.dot(self.alphas, svm_preds))


    # different number of classifiers
    def number_class(self, X):

        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        number = [] #array of predicted samples i.e. array of arrays

        for i in range(len(self.alphas)):
            number.append(self.alphas[i]*svm_preds[i])

        number = np.array(number)
        number = np.cumsum(number,axis=0)
        return np.sign(number)


# Diverse AdaBoostSVM
class Div_AdaBoostSVM(AdaBoostSVM):

    # Diversity threshold-constant and empty list
    eta = 0.74
    diversities = ([])
    Div_total = ([])

    def diversity(self, X, y):

        div = 0
        ensemble_pred = self.boost_predict(X)
        for i in range(len(y)):
            div += 1 if (y[i] != ensemble_pred[i]) else 0

        return div


    def svc_train(self, myKernel, myGamma, stepGamma, x_train, y_train, myWeights):

        if self.count == 0:
            myGamma = self.gammaIni

        while True:
            if myGamma<0:
                break

            errorOut = 0.0

            svcB = SVC(C = self.C, kernel='rbf', gamma=1/(2*(myGamma**2)), shrinking = True, probability = True, tol = 0.001)
            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)

            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            if self.count == 0:
                div = 0
                Div_threshold = -1
                Div_partial = 0

            else:
                div = self.diversity(x_train, y_pred)
                self.diversities = np.append(self.diversities, div)
                Div_partial = np.sum(self.diversities)/(len(y_train) * len(self.diversities))
                self.Div_total = np.append(self.Div_total, Div_partial)
                Div_threshold = self.eta * np.max(self.Div_total)


            #print(div/len(y_train), Div_partial, Div_threshold, len(self.diversities))

            # require an error below 50%, diversity above threshold and avoid null errors
            if errorOut < 0.5 and errorOut != 0 and Div_partial > Div_threshold:
                myGamma -= stepGamma
                break

            myGamma -= stepGamma

        return myGamma, errorOut, y_pred, svcB


# run the calculations
# get the data
#'titanic', 'two_norm', 'cancer', 'german', 'heart', 'solar','car','contra','nursery','tac_toe'
data = data_preparation()
sample = 'two_norm' # heart (issues); two_norm, nursery(large)
#X_train, Y_train, X_test, Y_test = data.dataset(sample, 0.4)

'''
# single support vector machine
weights= np.ones(len(Y_train))/len(Y_train)
svc = SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)
svc.fit(X_train, Y_train, weights)
Y_pred = svc.predict(X_test)
du.metrics(sample,'svm', svc, X_train, Y_train, Y_test, X_test, Y_pred)
'''
# comparison with other ml models (fit, predict and metrics)
#mc.comparison(sample, X_train, Y_train, Y_test, X_test)


n_class = ([])
div_class = ([])

normal_pre = ([])
normal_err = ([])
diverse_pre = ([])
diverse_err = ([])

for i in range(10):
    X_train, Y_train, X_test, Y_test = data.dataset(sample, 0.4)

    #AdaBoost support vector machine
    model1 = AdaBoostSVM(C = 150, gammaIni = 10)
    num_class = model1.fit(X_train, Y_train).count
    #y_preda = model1.boost_predict(X_test)
    n_class = np.append(n_class, num_class)

    test_pre = (model1.boost_predict(X_test) == Y_test).mean()
    test_err = (model1.boost_predict(X_test) != Y_test).mean()
    normal_pre = np.append(normal_pre, test_pre)
    normal_err = np.append(normal_err, test_err)

    #print('Normal AdaBoostSVM')
    #print(f'Test prec.: {test_pre:.1%}')
    #print(f'Test error: {test_err:.1%}')

    #test_number = model1.number_class(X_test)

    model2 = Div_AdaBoostSVM(C = 150, gammaIni = 10)
    num_class = model2.fit(X_train, Y_train).count
    #y_preda = model2.predict(X_test)
    div_class = np.append(div_class, num_class)

    test_pre = (model2.boost_predict(X_test) == Y_test).mean()
    test_err = (model2.boost_predict(X_test) != Y_test).mean()
    diverse_pre = np.append(diverse_pre, test_pre)
    diverse_err = np.append(diverse_err, test_err)

print('AdaBoost: ' + f'Test prec.: {np.mean(normal_pre)}, ' + f'Test error: {np.mean(normal_err)}')
print('Numero promedio de class: ', np.mean(n_class) )
print('Div_AdaBoostSVM' + f'Test prec.: {np.mean(diverse_pre)}, ' + f'Test error: {np.mean(diverse_err)}')
print('Numero promedio de class Div: ', np.mean(div_class) )

    #print('Diverse AdaBoostSVM')
    #print(f'Test prec.: {test_pre:.1%}')
    #print(f'Test error: {test_err:.1%}')

#test_number = model2.number_class(X_test)


'''
error = ([])
num = ([])
for i in range(len(test_number)):
    error_d = 0
    error_d = (test_number[i] != Y_test).mean()
    error   = np.append(error, [round(error_d * 100, 2)])
    num = np.append(num,i+1)

frame = pd.DataFrame(error,num)
print(frame)

import matplotlib.pyplot as plt
frame.plot()
plt.show()
'''
#du.cv_metrics(model, X_train, Y_train)

'''
run main function for every dataset
for item in sample_list:
    main(item)
to be pasted at the beginning of the svc_train function
# check normalization
check = 0.
for i in range(len(myWeights,)):
    check+=myWeights[i] #weights must add one, i.e. check=1.
'''
