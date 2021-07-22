'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# AdaBoost class
class AdaBoostSVM:

    def __init__(self, C, gammaEnd, myKernel, myDegree=1, myCoef0=1, Diversity=False, early_stop=False, debug=False, train_verbose=False):        
        self.C = C
        self.gammaEnd = gammaEnd
        self.myKernel = myKernel
        self.myDegree = myDegree
        self.myCoef0 = myCoef0
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors    = ([])
        self.precision = ([])
        self.train_scores = ([])
        self.test_scores = ([])
        self.count_over_train = ([])
        self.count_over_train_equal = ([])
        # Diversity threshold-constant and empty list
        self.m_div_flag = Diversity
        self.eta = 0.6
        self.diversities = ([])
        self.Div_total = ([])
        self.Div_partial = ([])
        self.debug = debug
        self.verbose_train = train_verbose
        self.early_flag = early_stop
        self.n_classifiers=0
        

    def svc_train(self, myGamma, stepGamma, x_train, y_train, myWeights, count, flag_div, value_div):

        if count == 0: myGamma = stepGamma

        while True:
            if myGamma > self.gammaEnd+stepGamma: return 0, 0, None, None

            errorOut = 0.0            
            svcB = SVC(C=self.C,
                       kernel=self.myKernel,
                       degree=self.myDegree,
                       coef0=self.myCoef0,
                       gamma=myGamma,                       #gamma=1/(2*(myGamma**2)),
                       shrinking=True,
                       probability=True,
                       tol=0.001,
                       cache_size=1000)
            
            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)
                                        
            # error calculation
            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            error_pass = errorOut < 0.49 and errorOut > 0.0
            # Diverse_AdaBoost, if Diversity=False, diversity plays no role in classifier selection
            div_pass, tres = self.pass_diversity(flag_div, value_div, count, error_pass)
            if(error_pass and not div_pass): value_div = self.diversity(x_train, y_pred, count)
            if self.debug: print('error_flag: %5s | div_flag: %5s | div_value: %5s | Threshold: %5s | no. data: %5s | count: %5s | error: %5.2f | gamma: %5.2f | diversities  %3s '
                  %(error_pass, div_pass, value_div, tres, len(y_pred), count, errorOut, myGamma, len(self.diversities)))
            
            # require an error below 50%, avoid null errors and diversity requirement
            if(error_pass and div_pass):
                #myGamma -= stepGamma
                break

            myGamma += stepGamma

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

        div_flag, div_value = self.m_div_flag, 0

        gammaMax = self.gammaEnd        
        gammaStep, gammaVar = gammaMax/100., 1/100.
        cost, count, norm = 1, 0, 0.0
        h_list = []

        # AdaBoost loop
        while True:
            if self.early_flag:
                if self.early_stop(count, X_test, Y_test, gammaVar): break  # early stop based on a score
            if count == 0:
                norm = 1.0
                new_weights = weights.copy()

            new_weights = new_weights/norm

            self.weights_list.append(new_weights)
            if self.debug : print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")

            # call svm, weight samples, iterate sigma(gamma), get errors, obtain predicted classifier (h as an array)
            gammaVar, error, h, learner = self.svc_train(gammaVar, gammaStep, X_train, Y_train, new_weights, count, div_flag, div_value)

            if(gammaVar > gammaMax or error <= 0):# or learner == None or h == None):
                break

            # count how many times SVM we add the ensemble
            count += 1
            self.n_classifiers +=1

            # calculate training precision
            fp,tp = 0,0
            for i in range(n):
                if(Y_train[i]!=h[i]): fp+=1
                else:                 tp+=1
        
            # store the predicted classes
            h_temp = h.tolist()
            h_list.append(h_temp)

            # store errors and precision
            self.errors = np.append(self.errors, [error])
            self.precision = np.append(self.precision, [tp / (tp + fp)])
            
            # calculate diversity
            if self.m_div_flag:
                div_value = self.diversity(X_train, h, count) # cf. h == y_pred
        
            # classifier weights (alpha), obtain and store
            x = (1 - error)/error
            alpha = 0.5 * np.log(x)
            self.alphas   = np.append(self.alphas, alpha)
            self.weak_svm = np.append(self.weak_svm, learner)

            # get training errors used for early stop
            # train_score = 1 - accuracy_score(Y_train, self.predict(X_train))
            # self.train_scores = np.append(self.train_scores, train_score)
            
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
            if gammaVar >= gammaMax:#) or (gammaVar < 0):
                break
            # end of adaboost loop

        print(count,'number of classifiers')
        self.n_classifiers = count
        if(count==0):
            print(' WARNING: No selected classifiers in the ensemble!!!!')
            # print('Adding artifically the first one with NO requirements!!!!!!')
            self.n_classifiers = 0
            # self.alphas   = np.append(self.alphas, 1.0)
            # # artificially added classifier, when no classifier has been
            # # selected, the first classifier
            # single = SVC(C=self.C,
            #              kernel=self.myKernel,
            #              degree=self.myDegree,
            #              coef0=self.myCoef0,
            #              gamma=1/(2*(self.gammaEnd**2)),
            #              shrinking=True,
            #              probability=True,
            #              tol=0.001,
            #              cache_size=10000 )
            # single.fit(X_train, Y_train)
            # self.weak_svm = np.append(self.weak_svm, single)
            return self

        # show the training the performance (optional)
        if(self.verbose_train):
            h_list = np.array(h_list)            
            # calculate the final classifier
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
                        
        return self


    def predict(self, X):
        # Make predictions using already fitted model
        # print(len(self.alphas), len(self.weak_svm), "how many alphas we have")
        # print(type(X.shape[0]), 'check size ada-boost')
        if self.n_classifiers == 0: return np.zeros(X.shape[0])
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        return np.sign(np.dot(self.alphas, svm_preds))
    

    def diversity(self, x_train, y_pred, count): # this function gets div for a single classifier 
        if count==1: return len(y_pred)/len(y_pred) # for 1st selected classifer, set max diversity
        div = 0
        ensemble_pred = self.predict(x_train) # uses the already selected classifiers in ensemble
        for i in range(len(y_pred)):
            if  (y_pred[i] != ensemble_pred[i]):  div += 1
            elif(y_pred[i] == ensemble_pred[i]):  div += 0
        return div/len(y_pred)
    

    def pass_diversity(self, flag_div, val_div, count, pass_error):
        threshold_div = 0
        if not flag_div:   return True, threshold_div
        if not pass_error: return True, threshold_div        
        if not count != 0: return True, threshold_div

        if(len(self.diversities)==0):
            self.diversities = np.append(self.diversities, val_div)
            return True, threshold_div
        else:
            div_ens = np.mean(np.append(self.diversities, val_div)) # d_ens=sum/t_cycles_accepted

        #print(self.diversities, val_div, div_ens, self.n_classifiers) # check behavoir diversity
        threshold_div = self.eta # self.eta * np.max(self.diversities)
        if div_ens >= threshold_div:
            self.diversities = np.append(self.diversities, val_div)                
            return True, threshold_div
        else:
            return False, threshold_div

        
    def early_stop(self, count, x_test, y_test, gammaVar):
        strip_length = 5
        if count == 0 or count%strip_length != 0:
            return False
        
        test_score = 1 - accuracy_score(y_test, self.predict(x_test))
        if len(self.test_scores) == 0:
            self.test_scores = np.append(self.test_scores, test_score)
            return False

        min_test_score = np.amin(self.test_scores) # stop if we reached perfect testing score
        if(min_test_score==0):
            print(min_test_score, 'min_test_score')
            return True
        
        self.test_scores = np.append(self.test_scores, test_score)

        # early stop definition 3 (see the paper)
        index_test = int(count/strip_length)
        current_error = self.test_scores[index_test-1]
        past_error = self.test_scores[index_test - int(strip_length/strip_length) - 1]
        if(current_error == past_error):
            self.count_over_train_equal = np.append(self.count_over_train_equal, 1)
        if(current_error > past_error):
            self.count_over_train = np.append(self.count_over_train, 1)
            self.count_over_train_equal = ([])
        if(current_error < past_error):
            self.count_over_train_equal = ([])

        counter_flag = count >= 100
        if(counter_flag):
            counter_flag = count >= 250
        # print('current:', round(current_error,2), ' past:',round(past_error,2), ' count:', count, ' length: ',
        #       len(self.count_over_train), 'another check', gammaVar, 'count equal:', len(self.count_over_train_equal))
        return len(self.count_over_train) >= 4 or counter_flag or len(self.count_over_train_equal) >= 15# previous_score <= test_score

            
    def early_stop_alternative(self, count, x_test, y_test, gammaVar): # not usable right now
        strip_length = 5
        if count == 0 or count%strip_length != 0:  return False
        # calculate the training strip progress and generalization (see paper)
        strip = self.train_scores[count - strip_length + 1 - 1:count]
        progress = 1000 * ( np.sum(strip)/(strip_length*np.amin(strip)) -1)
        gl = 100 * ( (test_score / min_test_score ) - 1)
        

    def _check_X_y(self, X, y):
        # Validate assumptions about format of input data. Expecting response variable to be formatted as ±1
        assert set(y) == {-1, 1} or set(y) == {-1} or set(y) == {1} # extra conditions for highly imbalance
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
                # print('check point: ', len(steps), len(decision))
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
        # clean is needed in case of running several times
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
        self.train_scores = ([])
        self.test_scores = ([])
        self.count_over_train = ([])
        self.count_over_train_equal = ([])
        self.n_classifiers=0


'''
# check normalization 
check = 0.
for i in range(len(myWeights,)):
    check+=myWeights[i] #weights must add one, i.e. check=1.
'''
