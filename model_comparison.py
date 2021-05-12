
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# model comparison module

# machine learning modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# import module for data utils
import data_utils as du
import pandas as pd
import datetime
import numpy as np

# perform the ml algos and generate report (fit , predict and metrics)
def comparison(sample, X_train, Y_train, Y_test, X_test):
    # initialise list to store process times
    times_list = ['Time']

    # creating the dataframe containing the output metrics
    metrics_list = ['Cross Validation', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_df = pd.DataFrame({'Metric': metrics_list})

    # support vector machine (single case)
    weights= np.ones(len(Y_train))/len(Y_train)
    svc = SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)
    start = datetime.datetime.now()
    svc.fit(X_train, Y_train, weights)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred = svc.predict(X_test)
    temp_list = du.metrics(sample, 'SVC', svc, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'SVC': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # RANDOM forest classifier
    random_forest = RandomForestClassifier(n_estimators=100)
    start = datetime.datetime.now()
    random_forest.fit(X_train, Y_train)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred = random_forest.predict(X_test)
    temp_list = du.metrics(sample, 'rForest',random_forest, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'Random Forest': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
    start = datetime.datetime.now()
    AdaBoost.fit(X_train, Y_train)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred = AdaBoost.predict(X_test)
    temp_list = du.metrics(sample, 'AdaBoost-SAMME', AdaBoost, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'AdaBoost': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # Neural Network Multi Layer Perceptron classifier
    NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)
    start = datetime.datetime.now()
    NeuralNet.fit(X_train, Y_train)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred = NeuralNet.predict(X_test)
    temp_list = du.metrics(sample, 'NeuralNet', NeuralNet, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'NeuralNet': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # Gradient Boost Classifier XGBoost
    model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    start = datetime.datetime.now()
    model_GBC.fit(X_train, Y_train)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred=model_GBC.predict(X_test)
    temp_list = du.metrics(sample, 'XGBoost', model_GBC, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'XGBoost': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)

    # K neighbors classifier. n_neighbors=3 because there are 2 classes
    knn = KNeighborsClassifier(n_neighbors=3)
    start = datetime.datetime.now()
    knn.fit(X_train, Y_train)
    end = datetime.datetime.now()
    times_list.append(end - start)
    Y_pred = knn.predict(X_test)
    temp_list = du.metrics(sample, 'KNN', knn, X_train, Y_train, Y_test, X_test, Y_pred)
    temp_df = pd.DataFrame({'KNN': temp_list})
    metrics_df = pd.concat([metrics_df, temp_df], axis = 1)
    metrics_df.loc[6] = times_list
    #print(times_list)

    metrics_df.to_csv('output/{}/metrics_report.csv'.format(sample), index=False)
