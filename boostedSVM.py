#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# python basics
import sys

# data analysis and wrangling
import numpy as np

# machine learning
from sklearn.svm import SVC, LinearSVC

# AdaBoost class

class AdaBoostSVM:

    def __init__(self, C, gammaIni):

        self.C = C
        self.gammaIni = gammaIni
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors    = ([])
        self.precision = ([])


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


    def svc_train(self, myKernel, myGamma, stepGamma, x_train, y_train, myWeights, count):

        if count == 0:
            myGamma = self.gammaIni

        if myGamma <= 0:
            return 0, 0, None, None

        while True:

            if myGamma <= 0:
                return 0, 0, None, None

            errorOut = 0.0

            svcB = SVC(
                    C=self.C,
                    kernel='rbf',
                    gamma=1/(2*(myGamma**2)),
                    shrinking=True,
                    probability=True,
                    tol=0.001,
                    cache_size=5000
                )

            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)

            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            # require an error below 50% and avoid null errors
            if(errorOut < 0.49 and errorOut > 0.0):
                #myGamma -= stepGamma
                break

            myGamma -= stepGamma

        return myGamma, errorOut, y_pred, svcB


    def fit(self, X, y):

        X_train, Y_train = self._check_X_y(X, y)
        n = X.shape[0]
        weights= np.ones(n)/n

        gammaMin, gammaStep, gammaVar = 1.0, 0.1, 0.0
        cost, count, norm = 1, 0, 0.0
        h_list = []

        # AdaBoost loop
        while True:
            if count == 0:
                norm = 1.0
                new_weights = weights.copy()

            new_weights = new_weights/norm

            self.weights_list.append(new_weights)

            # call svm, weight samples, iterate sigma(gamma), get errors, obtain predicted classifier (h as an array)
            gammaVar, error, h, learner = self.svc_train('rbf', gammaVar, gammaStep, X_train, Y_train, new_weights, count)

            if(gammaVar <= 0 or error <= 0):# or learner == None or h == None):
                break

            # count how many times SVM runs
            count += 1

            # calculate training precision
            fp,tp = 0,0
            for i in range(n):
                if(Y_train[i]!=h[i]): fp+=1
                else:                 tp+=1
        
        
            # store the predicted classes
            h_temp = h.tolist()
            h_list.append(h_temp)

            print("Error: {} Precision: {} Gamma: {} ".format(round(error,4), round(tp / (tp + fp),4), round(gammaVar+gammaStep,2)))
            # store errors
            self.errors = np.append(self.errors, [error])
            # store precision
            self.precision = np.append(self.precision, [tp / (tp + fp)])

            # classifier weights (alpha), obtain and store
            x = (1 - error)/error
            alpha = 0.5 * np.log(x)
            self.alphas   = np.append(self.alphas, alpha)
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
            if gammaVar <= gammaMin:#) or (gammaVar < 0):
                break

            # end of adaboot loop

        h_list = np.array(h_list)

        print(count,'number of classifiers')
        if(count==0):
            sys.exit('No classifiers in the ensemble, try again!')

        # start to calculate the final classifier
        h_alpha = np.array([h_list[i]*self.alphas[i] for i in range(count)])

        # final classifier is an array (size of number of data points)
        final = ([])
        for j in range(len(h_alpha[0])):
            suma = 0.0
            for i in range(count):
                suma+=h_alpha[i][j]
            final = np.append(final, [np.sign(suma)])

        # final precision calculation
        final_fp, final_tp  = 0,0
        for i in range(n):
            if(Y_train[i]!=final[i]):  final_fp+=1
            else:                      final_tp+=1

        final_precision = final_tp / (final_fp + final_tp)
        print("Final Precision: {} ".format( round(final_precision,4)) )
        #du.metrics(sample,'svmBoosted', svcB, X_train, Y_train, Y_test, X_test, Y_predB)

        return self


    def predict(self, X):
        # Make predictions using already fitted model
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        print(svm_preds.shape, 'Predict function')

        return np.sign(np.dot(self.alphas, svm_preds))

    
    def decision_thresholds(self, X, glob_dec):
        # function to threshold the svm decision, by varying the bias(intercept)
        svm_decisions = np.array([learner.decision_function(X) for learner in self.weak_svm])
        svm_biases    = np.array([learner.intercept_ for learner in self.weak_svm])

        thres_decision = []

        steps = np.linspace(-50,50,num=1001)
        decision,decision_temp = ([]),([])

        if not glob_dec: # threshold each individual classifier
            for i in range(len(steps)):
                decision = np.array([np.sign(svm_decisions[j] - svm_biases[j] + steps[i]*svm_biases[j]) for j in range(len(svm_biases))])
                thres_decision.append(decision)
                
            thres_decision = np.array(thres_decision)
            
            final_threshold_decisions = []
            for i in range(len(steps)):
                final = np.sign(np.dot(self.alphas,thres_decision[i]))
                final_threshold_decisions.append(final)
            
            return np.array(final_threshold_decisions)
        
        elif glob_dec: # glob_dec == true threshold the global final classifier
            decision = np.array([svm_decisions[j] + svm_biases[j] for j in range(len(svm_biases))])
            decision = np.dot(self.alphas,decision)

            for i in range(len(steps)):
                decision_temp = np.array([np.sign(decision[j] + steps[i] ) for j in range(len(svm_biases))]) #*svm_biases[j]
                thres_decision.append(decision_temp)
                        
            return np.array(thres_decision)
            
    # different number of classifiers
    def number_class(self, X):

        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        number = [] #array of predicted samples i.e. array of arrays

        for i in range(len(self.alphas)):
            number.append(self.alphas[i]*svm_preds[i])

        number = np.array(number)
        number = np.cumsum(number,axis=0)
        return np.sign(number)

    def get_metrics(self):
        return np.array(self.weights_list), self.errors, self.precision
      

# Diverse AdaBoostSVM
class Div_AdaBoostSVM(AdaBoostSVM):

    # Diversity threshold-constant and empty list
    eta = 0.74
    diversities = ([])
    Div_total = ([])

    def diversity(self, X, y):

        div = 0
        ensemble_pred = self.predict(X)
        for i in range(len(y)):
            div += 1 if (y[i] != ensemble_pred[i]) else 0

        return div


    def svc_train(self, myKernel, myGamma, stepGamma, x_train, y_train, myWeights, count):

        if count == 0:
            myGamma = self.gammaIni

        if myGamma <= 0:
            return 0, 0, None, None

        while True:
            if myGamma <= 0:
                return 0, 0, None, None

            errorOut = 0.0

            svcB = SVC(
                    C=self.C,
                    kernel='rbf',
                    gamma=1/(2*(myGamma**2)),
                    shrinking=True,
                    probability=True,
                    tol=0.001,
                    cache_size=5000)

            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)

            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            if count == 0:

                div = 0
                Div_threshold = -1
                Div_partial = 0

            else:

                div = self.diversity(x_train, y_pred)
                self.diversities = np.append(self.diversities, div)
                Div_partial = np.sum(self.diversities)/(len(y_train) * len(self.diversities))
                self.Div_total = np.append(self.Div_total, Div_partial)
                Div_threshold = self.eta * np.max(self.Div_total)

            # require an error below 49%, diversity above threshold and avoid null errors
            if errorOut < 0.49 and errorOut != 0 and Div_partial > Div_threshold:
                break

            myGamma -= stepGamma

        return myGamma, errorOut, y_pred, svcB

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

