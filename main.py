#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
"""

# main module
import sys
import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVC, LinearSVC # machine learning
from sklearn.metrics import accuracy_score,auc

# framework includes
from data_preparation import data_preparation
import data_utils as du
import model_maker as mm
import model_comparison as mc
import data_visualization as dv
from model_performance import model_performance
import stats_summary as ss
from genetic_selection import genetic_selection


from boostedSVM import AdaBoostSVM

if len(sys.argv) != 2:
    sys.exit("Provide data sample name. Try again!")

#states = "omega" # All, omega, cascades, sigmaLamb
sample_input = sys.argv[1]


# make directories
sample_list = ["titanic", "cancer", "german", "heart", "solar","car","contra","tac_toe", "belle2_i", "belle2_ii","belle_iii"]
du.make_directories(sample_list)

# kernel selection
kernel_list = ["linear", "poly", "rbf", "sigmoid"]
myKernel = "rbf"

# get the data
data = data_preparation()
sample_list = ["titanic"]
sample_list = [sample_input]

# loop over datasets in sample_list for AdaBoostSVM and other classifiers. get ROC curves & metrics
for name in sample_list:
    print("Analysing sample: ", name)
    split_flag = False
    if name=="belle2_iii": split_flag = True

    X_train, Y_train, X_test, Y_test = \
        data.dataset(sample_name=name,
                     sampling=False,split_sample=0.4,train_test=split_flag)


    # # run AdaBoost support vector machine
    # print("AdaBoost-support vector machines")
    # model = mm.adaboost_svm()
    # #model = AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel)

    # start = datetime.datetime.now()
    # model.fit(X_train, Y_train)
    # end = datetime.datetime.now()
    # elapsed_time = pd.DataFrame({"Elapsed time": [end - start]})

    # elapsed_time.to_csv("output/" + name +  "/" + "AdaBoostSVM_time.csv", index=False)
    # y_preda = model.predict(X_test)
    # print("Final test prediction:   ", accuracy_score(Y_test, y_preda))
    # y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
    # TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)

    # nWeaks = len(model.alphas) # print on plot no. classifiers
    # dv.plot_roc_curve(TPR,FPR,name,"sorted", glob_local=True, name="nom", kernel=myKernel, nClass=nWeaks)        
    # print("End adaboost")
    
    # # run Diverse-AdaBoost Diversity support vector machine
    # print("Diverse-AdaBoost-support vector machines")
    # #model_a = AdaBoostSVM(C=50, gammaIni=5, myKernel=myKernel, Diversity=True)
    # model_a = mm.adaboost_svm(True)
    # model_a.fit(X_train, Y_train)
    # nWeaks=len(model_a.alphas)
    # y_preda_a = model_a.predict(X_test)
    # y_thresholds_a = model_a.decision_thresholds(X_test, glob_dec=True)
    # TPR_a, FPR_a = du.roc_curve_adaboost(y_thresholds_a, Y_test)
    # area = auc(FPR_a,TPR_a)
    # print("Final test prediction:   ", accuracy_score(Y_test, y_preda_a), "AUC: ", area)

    # dv.plot_roc_curve(TPR_a,FPR_a,name,"sorted", glob_local=True, name="div", kernel=myKernel, nClass=nWeaks)
    # print("End adaboost")

    # test statistical results
    # ss.mcnemar_test(name, model="diverse", GA_score="acc", GA_selec="roulette", train_test=False)
    # ss.mcnemar_test(name, model="no_div",  GA_score="acc", GA_selec="roulette", train_test=False)

    # ss.stats_results(name, n_cycles=5, kfolds=3, n_reps=2, boot_kfold ="bootstrap", GA_score="acc", GA_selec="roulette")
    # ss.stats_results(name, n_cycles=5, kfolds=3, n_reps=2, boot_kfold ="kfold",     GA_score="acc", GA_selec="roulette")

    # test genetic selection
    model_test = AdaBoostSVM(C=150, gammaIni=10, myKernel="rbf", Diversity=False, early_stop=True, debug=False)

    start_GA = datetime.datetime.now()
    GA_selection = genetic_selection(model_test, "absv", X_train, Y_train, X_test, Y_test,
                                     pop_size=10, chrom_len=100, n_gen=50, coef=0.5, mut_rate=0.3, score_type="acc", selec_type="roulette")
    GA_selection.execute()
    GA_train_indexes = GA_selection.best_population()
    end_GA = datetime.datetime.now()
    elapsed_time_GA = end_GA - start_GA

    X_train_GA, Y_train_GA, X_test_GA, Y_test_GA = \
        data.dataset(sample_name=name, train_test=split_flag, indexes=GA_train_indexes)

    # compare between data inputs
    # traditional input
    start = datetime.datetime.now()
    model_test.fit(X_train, Y_train)
    y_preda = model_test.predict(X_test)
    y_thresholds = model_test.decision_thresholds(X_test, glob_dec=True)
    TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
    nWeaks = len(model_test.alphas) # print on plot no. classifiers
    dv.plot_roc_curve(TPR,FPR,name,"normal", glob_local=True, name="nom", kernel=myKernel, nClass=nWeaks)
    model_test.clean()
    end = datetime.datetime.now()
    elapsed_time = end - start
    print(len(X_train), len(X_test))
    print("Elapsed time TRADITIONAL = " + str(elapsed_time))

    # genetic selection input
    start = datetime.datetime.now()
    model_test.fit(X_train_GA, Y_train_GA)
    y_preda = model_test.predict(X_test_GA)
    y_thresholds = model_test.decision_thresholds(X_test_GA, glob_dec=True)
    TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test_GA)
    nWeaks = len(model_test.alphas) # print on plot no. classifiers
    dv.plot_roc_curve(TPR,FPR,name,"normal", glob_local=True, name="GA", kernel=myKernel, nClass=nWeaks)
    model_test.clean()
    end = datetime.datetime.now()
    elapsed_time = end - start
    print(len(X_train_GA), len(X_test_GA))
    print("Elapsed time GENETIC = " + str(elapsed_time))
    print("GENETIC selection time = " + str(elapsed_time_GA))


    # # do the statistical analysis of the performance across different models
    # ss.mcnemar_test(name, model="diverse", train_test=False)
    # ss.mcnemar_test(name, model="no_div",  train_test=False)

    # start = datetime.datetime.now()
    # bootstrap
    # ss.stats_results(name, n_cycles=50, kfolds=10, n_reps=10, boot_kfold ="bootstrap")
    # kfold cross-validation
    # ss.stats_results(name, n_cycles=5, kfolds=10, n_reps=10, boot_kfold ="kfold")
    # end = datetime.datetime.now()
    # elapsed_time = end - start
    # print("Elapsed total time GENETIC = " + str(elapsed_time))

        
#performance = model_performance(model, X_train, Y_train, X_test, Y_test)

# comparison with other ml models (fit, predict and metrics)
# mc.comparison(name, X_train, Y_train, Y_test, X_test)
# metrics (via cross-validation)
# du.cv_metrics(model, X_train, Y_train)
