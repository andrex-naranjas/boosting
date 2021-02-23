#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import numpy as np
from sklearn.metrics import accuracy_score,auc,precision_score,roc_auc_score,f1_score,recall_score
from sklearn.utils import resample # for bootstraping
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd,MultiComparison
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import norm, normaltest, shapiro, chisquare, kstest
import pandas as pd
from sklearn.model_selection import RepeatedKFold

# framework includes
from data_preparation import data_preparation
import data_utils as du
import model_maker as mm
import data_visualization as dv


def bootstrap(model, sample_name, isAB_SVM, n_cycles, train_test, split_frac=0.6):
    
    # fetch data_frame without preparation
    data = data_preparation()    
    if not train_test: sample_df = data.fetch_data(sample_name)
    else: sample_train_df, sample_test_df = data.fetch_data(sample_name)
        
    area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores = ([]),([]),([]),([]),([]),([])

    data_size = sample_df.shape[0]
    n_samples = int(split_frac*data_size)
    
    # bootstrap score calculations
    for _ in range(n_cycles): # arbitrary number of bootstrap samples to produce
        
        if not train_test:            
            sampled_data_train   = resample(sample_df, replace = True, n_samples = n_samples, random_state = None)
            # test data are the complement of full input data that is not considered for training
            sampled_train_no_dup = sampled_data_train.drop_duplicates(keep=False)
            sampled_data_test    = pd.concat([sample_df,sampled_train_no_dup]).drop_duplicates(keep=False)
            
            X_train, Y_train = data.dataset(sample_name=sample_name, data_set=sampled_data_train,
                                            sampling=True, split_sample=0.0, train_test=train_test)
            X_test, Y_test   = data.dataset(sample_name=sample_name, data_set=sampled_data_test,
                                            sampling=True, split_sample=0.0, train_test=train_test)
            
        else:
            sampled_data_train = resample(sample_train_df, replace = True, n_samples = 5000,  random_state = 0)
            sampled_data_test  = resample(sample_test_df,  replace = True, n_samples = 10000, random_state = 0)
            X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, data_set='',
                                                            data_train=sampled_data_train, data_test = sampled_data_test,
                                                            sampling=True, split_sample=0.4, train_test=True)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        prec = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        acc = accuracy_score(Y_test, y_pred)
        gmean = np.sqrt(prec*recall)

        if isAB_SVM:
            y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
            area = auc(FPR,TPR)
            model.clean()
        else:
            Y_pred_prob = model.predict_proba(X_test)[:,1]            
            area = roc_auc_score(Y_test, Y_pred_prob)
        
        area_scores   = np.append(area_scores, area)
        prec_scores   = np.append(prec_scores, prec)
        f1_scores     = np.append(f1_scores,   f1)
        recall_scores = np.append(recall_scores, recall)
        acc_scores    = np.append(acc_scores, acc)
        gmean_scores  = np.append(gmean_scores, gmean)

    return area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores


def cross_validation(model, sample_name, isAB_SVM, kfolds, n_reps, train_test):
    
    # fetch data
    data = data_preparation()
    if not train_test: sample_df = data.fetch_data(sample_name)
    else: sample_train_df, sample_test_df = data.fetch_data(sample_name)
    X,Y = data.dataset(sample_name=sample_name, data_set=sample_df,
                       sampling=True, split_sample=0.0, train_test=train_test)
    
    X,Y = X.values, Y.values

    area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores = ([]),([]),([]),([]),([]),([])
    
    # n-k fold cross validation, n_cycles = n_splits * n_repeats
    rkf = RepeatedKFold(n_splits = kfolds, n_repeats = n_reps, random_state = None)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)                
        y_pred = model.predict(X_test)
        prec = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        acc = accuracy_score(Y_test, y_pred)
        gmean = np.sqrt(prec*recall)

        if isAB_SVM:
            y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
            area = auc(FPR,TPR)
            model.clean()
        else:
            Y_pred_prob = model.predict_proba(X_test)[:,1]            
            area = roc_auc_score(Y_test, Y_pred_prob)
        
        area_scores   = np.append(area_scores, area)
        prec_scores   = np.append(prec_scores, prec)
        f1_scores     = np.append(f1_scores,   f1)
        recall_scores = np.append(recall_scores, recall)
        acc_scores    = np.append(acc_scores, acc)
        gmean_scores  = np.append(gmean_scores, gmean)

    return area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores


def anova_test(a,b,c,d,e,f,g):
    print('ANOVA TEEEST!')
    fvalue, pvalue = f_oneway(a, b, c, d, e, f, g)
    print(fvalue, pvalue)

    
def tukey_test(a,b,c,d,e,f,g):
    print('TUUUKEY TEEEST!')
    index_a = np.array([ (int(1)) for i in range(len(a))])
    index_b = np.array([ (int(2)) for i in range(len(b))])
    index_c = np.array([ (int(3)) for i in range(len(c))])
    index_d = np.array([ (int(4)) for i in range(len(d))])
    index_e = np.array([ (int(5)) for i in range(len(e))])
    index_f = np.array([ (int(6)) for i in range(len(f))])
    index_g = np.array([ (int(7)) for i in range(len(g))])
    
    indexes= np.concatenate((index_a, index_b, index_c, index_d, index_e, index_f, index_g), axis=0)
    values = np.concatenate((a, b, c, d, e, f, g), axis=0)
    data   = {'means':values, 'group':indexes}

    MultiComp2 = MultiComparison(data['means'], data['group'])
    print(MultiComp2.tukeyhsd(0.05).summary())
    return MultiComp2.tukeyhsd(0.05)
    

def normal_test(sample,alpha,verbose):
    # hypothesis test: null hypothesis, the data is gaussian distributed

    # Shapiro-Wilk
    stat, p = shapiro(sample)
    if verbose:
        if p > alpha: print('Shapiro this is Gaussian', p)
        else:         print('Shapiro this is NOT Gaussian', p)

    # chisquare
    stat, p = chisquare(sample)
    if verbose:
        if p > alpha: print('Chisquare this is Gaussian', p)
        else:         print('Chisquare this is NOT Gaussian', p)

    # lilliefors
    stat, p = lilliefors(sample)
    if verbose:
        if p > alpha: print('Lilliefors this is Gaussian', p)
        else:         print('Lilliefors this is NOT Gaussian', p)

    # kolmogorov
    stat, p = kstest(sample, 'norm')
    if verbose:
        if p > alpha: print('Kolmogorov this is Gaussian', p)
        else:         print('Kolmogorov this is NOT Gaussian', p)

    # Angostino
    k2, p = normaltest(sample)
    if verbose:
        if p > alpha: print('Angostino this is Gaussian', p)
        else:         print('Angostino this is NOT Gaussian', p)    

    return p,alpha


def stats_results(name, n_cycles, kfolds, n_reps, boot_kfold ='', split_frac=0.6):
    
    methodName = ''
    if boot_kfold == 'bootstrap':
        methodName = 'boostrap'
        auc_svm_ab, prc_svm_ab, f1_svm_ab, rec_svm_ab, acc_svm_ab, gmn_svm_ab  = bootstrap(model=mm.adaboost_svm(False),sample_name=name, isAB_SVM=True,  n_cycles=n_cycles, train_test=False)
        auc_svm_abd,prc_svm_abd,f1_svm_abd,rec_svm_abd,acc_svm_abd,gmn_svm_abd = bootstrap(model=mm.adaboost_svm(True), sample_name=name, isAB_SVM=True,  n_cycles=n_cycles, train_test=False)
        auc_svm,    prc_svm,    f1_svm,    rec_svm,    acc_svm,    gmn_svm     = bootstrap(model=mm.single_svm(),       sample_name=name, isAB_SVM=False, n_cycles=n_cycles, train_test=False)
        auc_rand,   prc_rand,   f1_rand,   rec_rand,   acc_rand,   gmn_rand    = bootstrap(model=mm.rand_forest(),      sample_name=name, isAB_SVM=False, n_cycles=n_cycles, train_test=False)
        auc_bdt,    prc_bdt,    f1_bdt,    rec_bdt,    acc_bdt,    gmn_bdt     = bootstrap(model=mm.bdt_forest(),       sample_name=name, isAB_SVM=False, n_cycles=n_cycles, train_test=False)
        auc_neur,   prc_neur,   f1_neur,   rec_neur,   acc_neur,   gmn_neur    = bootstrap(model=mm.neural_net(),       sample_name=name, isAB_SVM=False, n_cycles=n_cycles, train_test=False)
        auc_knn,    prc_knn,    f1_knn,    rec_knn,    acc_knn,    gmn_knn     = bootstrap(model=mm.k_neighbors(),      sample_name=name, isAB_SVM=False, n_cycles=n_cycles, train_test=False)
        # to-do: add more classifiers
    elif boot_kfold == 'kfold':
        methodName   = 'kfold'
        auc_svm_ab, prc_svm_ab, f1_svm_ab, rec_svm_ab, acc_svm_ab, gmn_svm_ab  = cross_validation(model=mm.adaboost_svm(False),sample_name=name, isAB_SVM=True,  kfolds=10, n_reps=5, train_test=False)
        auc_svm_abd,prc_svm_abd,f1_svm_abd,rec_svm_abd,acc_svm_abd,gmn_svm_abd = cross_validation(model=mm.adaboost_svm(True), sample_name=name, isAB_SVM=True,  kfolds=10, n_reps=5, train_test=False)
        auc_svm,    prc_svm,    f1_svm,    rec_svm,    acc_svm,    gmn_svm     = cross_validation(model=mm.single_svm(),       sample_name=name, isAB_SVM=False, kfolds=10, n_reps=5, train_test=False)
        auc_rand,   prc_rand,   f1_rand,   rec_rand,   acc_rand,   gmn_rand    = cross_validation(model=mm.rand_forest(),      sample_name=name, isAB_SVM=False, kfolds=10, n_reps=5, train_test=False)
        auc_bdt,    prc_bdt,    f1_bdt,    rec_bdt,    acc_bdt,    gmn_bdt     = cross_validation(model=mm.bdt_forest(),       sample_name=name, isAB_SVM=False, kfolds=10, n_reps=5, train_test=False)
        auc_neur,   prc_neur,   f1_neur,   rec_neur,   acc_neur,   gmn_neur    = cross_validation(model=mm.neural_net(),       sample_name=name, isAB_SVM=False, kfolds=10, n_reps=5, train_test=False)
        auc_knn,    prc_knn,    f1_knn,    rec_knn,    acc_knn,    gmn_knn     = cross_validation(model=mm.k_neighbors(),      sample_name=name, isAB_SVM=False, kfolds=10, n_reps=5, train_test=False)

        
    mean_auc  = np.array([np.mean(auc_svm_ab), np.mean(auc_svm_abd), np.mean(auc_svm),  np.mean(auc_rand),   
                         np.mean(auc_bdt),     np.mean(auc_neur),    np.mean(auc_knn)])
    
    tukey_auc = tukey_test(auc_svm_ab, auc_svm_abd, auc_svm, auc_rand,
                              auc_bdt,    auc_neur,    auc_knn)

    mean_prc  = np.array([np.mean(prc_svm_ab), np.mean(prc_svm_abd), np.mean(prc_svm), np.mean(prc_rand),   
                          np.mean(prc_bdt),     np.mean(prc_neur),    np.mean(prc_knn)])
    
    tukey_prc = tukey_test(prc_svm_ab, prc_svm_abd, prc_svm, prc_rand,
                              prc_bdt,    prc_neur,    prc_knn)

    mean_f1   = np.array([np.mean(f1_svm_ab), np.mean(f1_svm_abd), np.mean(f1_svm), np.mean(f1_rand),   
                          np.mean(f1_bdt),    np.mean(f1_neur),    np.mean(f1_knn)])

    tukey_f1  = tukey_test(f1_svm_ab, f1_svm_abd, f1_svm, f1_rand,   
                              f1_bdt,    f1_neur,    f1_knn)

    mean_rec  = np.array([np.mean(rec_svm_ab), np.mean(rec_svm_abd), np.mean(rec_svm), np.mean(rec_rand),   
                          np.mean(rec_bdt),     np.mean(rec_neur),    np.mean(rec_knn)])
    
    tukey_rec = tukey_test(rec_svm_ab, rec_svm_abd, rec_svm, rec_rand,
                              rec_bdt,    rec_neur,    rec_knn)

    mean_acc  = np.array([np.mean(acc_svm_ab), np.mean(acc_svm_abd), np.mean(acc_svm), np.mean(acc_rand),   
                          np.mean(acc_bdt),     np.mean(acc_neur),    np.mean(acc_knn)])
    
    tukey_acc = tukey_test(acc_svm_ab, acc_svm_abd, acc_svm, acc_rand,
                              acc_bdt,    acc_neur,    acc_knn)

    mean_gmn  = np.array([np.mean(gmn_svm_ab), np.mean(gmn_svm_abd), np.mean(gmn_svm), np.mean(gmn_rand),   
                          np.mean(gmn_bdt),     np.mean(gmn_neur),    np.mean(gmn_knn)])
    
    tukey_gmn = tukey_test(gmn_svm_ab, gmn_svm_abd, gmn_svm, gmn_rand,
                              gmn_bdt,    gmn_neur,    gmn_knn)

    # latex tables
    f_tukey_noDiv = open('./tables/tukey_'+name+'_'+methodName+'_noDiv.tex', "w")
    dv.latex_table_tukey(False, mean_auc, tukey_auc, mean_prc,  tukey_prc, mean_f1,  tukey_f1,
                         mean_rec,  tukey_rec, mean_acc,  tukey_acc, mean_gmn,  tukey_gmn,  f_tukey_noDiv)
    f_tukey_noDiv.close()

    f_tukey_div = open('./tables/tukey_'+name+'_'+methodName+'_div.tex', "w")
    dv.latex_table_tukey(True, mean_auc, tukey_auc, mean_prc,  tukey_prc, mean_f1,  tukey_f1,
                         mean_rec,  tukey_rec, mean_acc,  tukey_acc, mean_gmn,  tukey_gmn,  f_tukey_div)
    f_tukey_div.close()



    # p_gaus, alpha = ss.normal_test(sample=auc_svm_boost,     alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_svm_boost_div, alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_svm_single,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_ran_forest,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_bdt_forest,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_net_neural,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(sample=auc_knn_neighb,    alpha=0.05, verbose=True)

    # p_gaus, alpha = ss.normal_test(prc_svm_boost,     alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_svm_boost_div, alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_svm_sigle,     alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_ran_forest,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_bdt_forest,    alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_net_neutral,   alpha=0.05, verbose=True)
    # p_gaus, alpha = ss.normal_test(prc_knn_neighb,    alpha=0.05, verbose=True)

    
    # ss.anova_test(auc_svm_boost,     
    #               auc_svm_boost_div, 
    #               auc_svm_single,    
    #               auc_ran_forest,    
    #               auc_bdt_forest,    
    #               auc_net_neural,    
    #               auc_knn_neighb)
        
    # ss.anova_test(prc_svm_boost,     
    #               prc_svm_boost_div,
    #               prc_svm_sigle,
    #               prc_ran_forest,    
    #               prc_bdt_forest,   
    #               prc_net_neutral,   
    #               prc_knn_neighb)
