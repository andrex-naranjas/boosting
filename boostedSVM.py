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
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# AdaBoost class

class AdaBoostSVM:

    def __init__(self, C, gammaIni, myKernel, Diversity=False, early_stop=False, debug=False):

        self.C = C
        self.gammaIni = gammaIni
        self.myKernel = myKernel
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors    = ([])
        self.precision = ([])
        self.train_scores = ([])
        self.test_scores = ([])
        # Diversity threshold-constant and empty list
        self.div_flag = Diversity
        self.eta = 0.5
        self.diversities = ([])
        self.Div_total = ([])
        self.Div_partial = ([])
        self.debug = debug
        self.early_flag = early_stop
        self.count_warning = 1
        

    def svc_train(self, myGamma, stepGamma, x_train, y_train, myWeights, count, flag_div, value_div):

        if count == 0: myGamma = self.gammaIni

        if myGamma <= 0:  return 0, 0, None, None

        while True:
            if myGamma <= 0: return 0, 0, None, None

            errorOut = 0.0
            
            svcB = SVC(C=self.C,
                    kernel=self.myKernel,
                    degree=1,
                    coef0=1,
                    gamma=1/(2*(myGamma**2)),
                    shrinking=True,
                    probability=True,
                    tol=0.001,
                    cache_size=10000
                )

            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)
                                        
            # error calculation
            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            error_pass = errorOut < 0.49 and errorOut > 0.0
            # Diverse_AdaBoost, if Diversity=False, diversity plays no role in classifier selection
            div_pass,tres = self.pass_diversity(flag_div, value_div, count, error_pass)
            if(error_pass and not div_pass): value_div = self.diversity(x_train, y_pred, count)
            if self.debug: print('error_flag: %5s | div_flag: %5s | div_value: %5s | Threshold: %5s | no. data: %5s | count: %5s | error: %5.2f | gamma: %5.2f | diversities  %3s '
                  %(error_pass, div_pass, value_div, tres, len(y_pred), count, errorOut, myGamma, len(self.diversities)))
            
            # require an error below 50%, avoid null errors and diversity requirement
            if(error_pass and div_pass):
                #myGamma -= stepGamma
                break

            myGamma -= stepGamma

        return myGamma, errorOut, y_pred, svcB


    def fit(self, X, y):

        if self.early_flag:
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
            X_train, Y_train = self._check_X_y(X_train, Y_train)
            X_test, Y_test = self._check_X_y(X_test, Y_test)
        else:
            X_train, Y_train = self._check_X_y(X, y)
                        
        n = X_train.shape[0]
        weights = np.ones(n)/n

        div_flag = self.div_flag
        div_value = 0
        
        gammaMin, gammaStep, gammaVar = 0.1, 0.1, 0.0
        cost, count, norm = 1, 0, 0.0
        h_list = []

        # AdaBoost loop
        while True:
            if self.early_flag:
                if self.early_stop(count, X_train, Y_train, X_test, Y_test): break  # early stop based on a score
            if count > 200: break
            if count == 0:
                norm = 1.0
                new_weights = weights.copy()

            new_weights = new_weights/norm

            self.weights_list.append(new_weights)
            if self.debug : print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")

            # call svm, weight samples, iterate sigma(gamma), get errors, obtain predicted classifier (h as an array)
            gammaVar, error, h, learner = self.svc_train(gammaVar, gammaStep, X_train, Y_train, new_weights, count, div_flag, div_value)

            if(gammaVar <= 0 or error <= 0):# or learner == None or h == None):
                break

            # count how many times SVM we add the ensemble
            count += 1

            # calculate training precision
            fp,tp = 0,0
            for i in range(n):
                if(Y_train[i]!=h[i]): fp+=1
                else:                 tp+=1
        
            # store the predicted classes
            h_temp = h.tolist()
            h_list.append(h_temp)

            # print("Error: {} Precision: {} Gamma: {} ".format(round(error,4), round(tp / (tp + fp),4), round(gammaVar+gammaStep,2)))
            # store errors
            self.errors = np.append(self.errors, [error])
            # store precision
            self.precision = np.append(self.precision, [tp / (tp + fp)])
            
            # calculate diversity
            div_value = self.diversity(X_train, h, count) # cf. h == y_pred
        
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

        print(count,'number of classifiers')
        self.count_warning == count
        if(count==0):
            print(' WARNING: No classifiers in the ensemble!')
            self.count_warning = 0
            return self

        h_list = np.array(h_list)
        
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
        print("Final training precision: {} ".format( round(final_precision,4)) )
        #du.metrics(sample,'svmBoosted', svcB, X_train, Y_train, Y_test, X_test, Y_predB)

        return self


    def predict(self, X):
        # Make predictions using already fitted model
        # print(len(self.alphas), len(self.weak_svm), "how many alphas we have")
        # print(type(X.shape[0]), 'check size ada-boost' )
        if self.count_warning == 0: return np.zeros(X.shape[0])
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        return np.sign(np.dot(self.alphas, svm_preds))
    

    def diversity(self, x_train, y_pred, count):
        if count==1: return len(y_pred) # for first selected classifer, set max diversity
        div = 0
        ensemble_pred = self.predict(x_train) # uses the already selected classifiers in ensemble
        for i in range(len(y_pred)):
            if  (y_pred[i] != ensemble_pred[i]):  div += 1
            elif(y_pred[i] == ensemble_pred[i]):  div += 0                        
        return div
    

    def pass_diversity(self, flag_div, val_div, count, pass_error):
        threshold_div = 0
        if not flag_div:   return True, threshold_div
        if not pass_error: return True, threshold_div        
        if not count != 0: return True, threshold_div

        if(len(self.diversities)==0):
            self.diversities = np.append(self.diversities, val_div)
        
        threshold_div = self.eta * np.max(self.diversities)
        
        if val_div >= threshold_div:
            self.diversities = np.append(self.diversities, val_div)                
            return True, threshold_div
        else:
            return False, threshold_div

        
    def early_stop(self, count, x_train, y_train, x_test, y_test):
        if count==0 or count%10 != 0:
            return False
        else:
            train_score = accuracy_score(y_train, self.predict(x_train))
            test_score = accuracy_score(y_test, self.predict(x_test))
            self.train_scores = np.append(self.train_scores, train_score)
            self.test_scores  = np.append(self.test_scores , test_score)
                        
            length = len(self.test_scores)            
            previous_score = self.test_scores[length - 2]

            if (previous_score < test_score):
                return True
            else:
                return False        
    

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
        
    
    def decision_thresholds(self, X, glob_dec):
        # function to threshold the svm decision, by varying the bias(intercept)
        svm_decisions = np.array([learner.decision_function(X) for learner in self.weak_svm])
        svm_biases    = np.array([learner.intercept_ for learner in self.weak_svm])

        thres_decision = []

        steps = np.linspace(-10,10,num=101)
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
                # print('daaaaaaaaaaaaaaaaaaaaaliiiii: ', len(steps), len(decision))
                decision_temp = np.array([np.sign(decision[j] + steps[i] ) for j in range(len(decision))]) #*svm_biases[j]
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

    
    def clean(self):
        # clean in case running several times is needed
        # when creating only one instance 
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors    = ([])
        self.precision = ([])
        self.eta = 0.5
        self.diversities = ([])
        self.Div_total = ([])
        self.Div_partial = ([])
        self.count_warning = 1


'''
# check normalization 
check = 0.
for i in range(len(myWeights,)):
    check+=myWeights[i] #weights must add one, i.e. check=1.
'''


    # def pass_diversity(self, flag_div, val_div, y_pred, count):

    #     if not flag_div: return True
        
    #     if count == 0: return True
    #     else:
    #         self.diversities = np.append(self.diversities, val_div)
    #         self.Div_partial = np.sum(self.diversities)/(len(y_pred) * len(self.diversities))                
    #         self.Div_total = np.append(self.Div_total, self.Div_partial)
    #         self.Div_threshold = self.eta * np.max(self.Div_total)
            
    #     # # if(self.Div_threshold > 0 and self.Div_partial > 0):
    #     # if(count!=0):
    #     #     print("Local value: ", val_div, len(y_pred), self.Div_partial, self.Div_threshold, count, "Diversity check", self.Div_partial > self.Div_threshold)#, "ratio: ", self.Div_partial/self.Div_threshold)

    #     return self.Div_partial > self.Div_threshold
