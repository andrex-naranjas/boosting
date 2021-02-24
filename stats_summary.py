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


def bootstrap(model, sample_name, roc_area, n_cycles, train_test=False, split_frac=0.6):
    
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

        if roc_area=="absv":
            y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
            area = auc(FPR,TPR)
            model.clean()
        elif roc_area=="prob":
            Y_pred_prob = model.predict_proba(X_test)[:,1]            
            area = roc_auc_score(Y_test, Y_pred_prob)
        elif roc_area=="deci":
            Y_pred_dec = model.decision_function(X_test)
            area = roc_auc_score(Y_test, Y_pred_dec)
            
        
        area_scores   = np.append(area_scores, area)
        prec_scores   = np.append(prec_scores, prec)
        f1_scores     = np.append(f1_scores,   f1)
        recall_scores = np.append(recall_scores, recall)
        acc_scores    = np.append(acc_scores, acc)
        gmean_scores  = np.append(gmean_scores, gmean)

    return area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores


def cross_validation(model, sample_name, roc_area, kfolds, n_reps, train_test=False):
    
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
        

        if roc_area=="absv":
            y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
            area = auc(FPR,TPR)
            model.clean()
        elif roc_area=="prob":
            Y_pred_prob = model.predict_proba(X_test)[:,1]            
            area = roc_auc_score(Y_test, Y_pred_prob)
        elif roc_area=="deci":
            Y_pred_dec = model.decision_function(X_test)
            area = roc_auc_score(Y_test, Y_pred_dec)

        
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

    
def tukey_test(c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,
               c11,c12,c13,c14,c15,c16,c17,c18,c19,c20):

    index_c01 = np.array([  (int(1)) for i in range(len(c01))])
    index_c02 = np.array([  (int(2)) for i in range(len(c02))])
    index_c03 = np.array([  (int(3)) for i in range(len(c03))])
    index_c04 = np.array([  (int(4)) for i in range(len(c04))])
    index_c05 = np.array([  (int(5)) for i in range(len(c05))])
    index_c06 = np.array([  (int(6)) for i in range(len(c06))])
    index_c07 = np.array([  (int(7)) for i in range(len(c07))])
    index_c08 = np.array([  (int(8)) for i in range(len(c08))])
    index_c09 = np.array([  (int(9)) for i in range(len(c09))])
    index_c10 = np.array([ (int(10)) for i in range(len(c10))])
    index_c11 = np.array([ (int(11)) for i in range(len(c11))])
    index_c12 = np.array([ (int(12)) for i in range(len(c12))])
    index_c13 = np.array([ (int(13)) for i in range(len(c13))])
    index_c14 = np.array([ (int(14)) for i in range(len(c14))])
    index_c15 = np.array([ (int(15)) for i in range(len(c15))])
    index_c16 = np.array([ (int(16)) for i in range(len(c16))])
    index_c17 = np.array([ (int(17)) for i in range(len(c17))])
    index_c18 = np.array([ (int(18)) for i in range(len(c18))])
    index_c19 = np.array([ (int(19)) for i in range(len(c19))])
    index_c20 = np.array([ (int(20)) for i in range(len(c20))])

    indexes= np.concatenate((index_c01,index_c02,index_c03,index_c04,index_c05,index_c06,index_c07,index_c08,index_c09,index_c10,
                             index_c11,index_c12,index_c13,index_c14,index_c15,index_c16,index_c17,index_c18,index_c19,index_c20), axis=0)
                            
    values = np.concatenate((c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,
                             c11,c12,c13,c14,c15,c16,c17,c18,c19,c20), axis=0)
                            
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

    if boot_kfold == 'bootstrap':        
        auc_svm_ab, prc_svm_ab, f1_svm_ab, rec_svm_ab, acc_svm_ab, gmn_svm_ab  = bootstrap(model=mm.adaboost_svm(False),sample_name=name, roc_area="absv", n_cycles=n_cycles)
        auc_svm_abd,prc_svm_abd,f1_svm_abd,rec_svm_abd,acc_svm_abd,gmn_svm_abd = bootstrap(model=mm.adaboost_svm(True), sample_name=name, roc_area="absv", n_cycles=n_cycles)
        auc_svm,    prc_svm,    f1_svm,    rec_svm,    acc_svm,    gmn_svm     = bootstrap(model=mm.single_svm(),       sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_svm_lin,prc_svm_lin,f1_svm_lin,rec_svm_lin,acc_svm_lin,gmn_svm_lin = bootstrap(model=mm.linear_svm(),       sample_name=name, roc_area="deci", n_cycles=n_cycles)
        auc_svm_bdt,prc_svm_bdt,f1_svm_bdt,rec_svm_bdt,acc_svm_bdt,gmn_svm_bdt = bootstrap(model=mm.bdt_svm(),          sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_svm_bag,prc_svm_bag,f1_svm_bag,rec_svm_bag,acc_svm_bag,gmn_svm_bag = bootstrap(model=mm.bag_svm(),          sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_ran,    prc_ran,    f1_ran,    rec_ran,    acc_ran,    gmn_ran     = bootstrap(model=mm.rand_forest(),      sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_bdt,    prc_bdt,    f1_bdt,    rec_bdt,    acc_bdt,    gmn_bdt     = bootstrap(model=mm.bdt_forest(),       sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_bag,    prc_bag,    f1_bag,    rec_bag,    acc_bag,    gmn_bag     = bootstrap(model=mm.bag_forest(),       sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_grad,   prc_grad,   f1_grad,   rec_grad,   acc_grad,   gmn_grad    = bootstrap(model=mm.grad_forest(),      sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_neur,   prc_neur,   f1_neur,   rec_neur,   acc_neur,   gmn_neur    = bootstrap(model=mm.neural_net(),       sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_knn,    prc_knn,    f1_knn,    rec_knn,    acc_knn,    gmn_knn     = bootstrap(model=mm.k_neighbors(),      sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_lin_dc, prc_lin_dc, f1_lin_dc, rec_lin_dc, acc_lin_dc, gmn_lin_dc  = bootstrap(model=mm.linear_dis(),       sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_quad,   prc_quad,   f1_quad,   rec_quad,   acc_quad,   gmn_quad    = bootstrap(model=mm.quad_dis(),         sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_gaus_nb,prc_gaus_nb,f1_gaus_nb,rec_gaus_nb,acc_gaus_nb,gmn_gaus_nb = bootstrap(model=mm.gauss_nb(),         sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_gaus_pc,prc_gaus_pc,f1_gaus_pc,rec_gaus_pc,acc_gaus_pc,gmn_gaus_pc = bootstrap(model=mm.gauss_pc(),         sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_log_reg,prc_log_reg,f1_log_reg,rec_log_reg,acc_log_reg,gmn_log_reg = bootstrap(model=mm.log_reg(),          sample_name=name, roc_area="prob", n_cycles=n_cycles)
        auc_ridge,  prc_ridge,  f1_ridge,  rec_ridge,  acc_ridge,  gmn_ridge   = bootstrap(model=mm.ridge_class(),      sample_name=name, roc_area="deci", n_cycles=n_cycles)
        auc_sgdc,   prc_sgdc,   f1_sgdc,   rec_sgdc,   acc_sgdc,   gmn_sgdc    = bootstrap(model=mm.sgdc_class(),       sample_name=name, roc_area="deci", n_cycles=n_cycles)
        auc_pass,   prc_pass,   f1_pass,   rec_pass,   acc_pass,   gmn_pass    = bootstrap(model=mm.pass_agre(),        sample_name=name, roc_area="deci", n_cycles=n_cycles)            
    elif boot_kfold == 'kfold':
        auc_svm_ab, prc_svm_ab, f1_svm_ab, rec_svm_ab, acc_svm_ab, gmn_svm_ab  = cross_validation(model=mm.adaboost_svm(False),sample_name=name, roc_area="absv", kfolds=kfolds, n_reps=n_reps)
        auc_svm_abd,prc_svm_abd,f1_svm_abd,rec_svm_abd,acc_svm_abd,gmn_svm_abd = cross_validation(model=mm.adaboost_svm(True), sample_name=name, roc_area="absv", kfolds=kfolds, n_reps=n_reps)
        auc_svm,    prc_svm,    f1_svm,    rec_svm,    acc_svm,    gmn_svm     = cross_validation(model=mm.single_svm(),       sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_svm_lin,prc_svm_lin,f1_svm_lin,rec_svm_lin,acc_svm_lin,gmn_svm_lin = cross_validation(model=mm.linear_svm(),       sample_name=name, roc_area="deci", kfolds=kfolds, n_reps=n_reps)
        auc_svm_bdt,prc_svm_bdt,f1_svm_bdt,rec_svm_bdt,acc_svm_bdt,gmn_svm_bdt = cross_validation(model=mm.bdt_svm(),          sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_svm_bag,prc_svm_bag,f1_svm_bag,rec_svm_bag,acc_svm_bag,gmn_svm_bag = cross_validation(model=mm.bag_svm(),          sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_ran,    prc_ran,    f1_ran,    rec_ran,    acc_ran,    gmn_ran     = cross_validation(model=mm.rand_forest(),      sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_bdt,    prc_bdt,    f1_bdt,    rec_bdt,    acc_bdt,    gmn_bdt     = cross_validation(model=mm.bdt_forest(),       sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_bag,    prc_bag,    f1_bag,    rec_bag,    acc_bag,    gmn_bag     = cross_validation(model=mm.bag_forest(),       sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_grad,   prc_grad,   f1_grad,   rec_grad,   acc_grad,   gmn_grad    = cross_validation(model=mm.grad_forest(),      sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_neur,   prc_neur,   f1_neur,   rec_neur,   acc_neur,   gmn_neur    = cross_validation(model=mm.neural_net(),       sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_knn,    prc_knn,    f1_knn,    rec_knn,    acc_knn,    gmn_knn     = cross_validation(model=mm.k_neighbors(),      sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_lin_dc, prc_lin_dc, f1_lin_dc, rec_lin_dc, acc_lin_dc, gmn_lin_dc  = cross_validation(model=mm.linear_dis(),       sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_quad,   prc_quad,   f1_quad,   rec_quad,   acc_quad,   gmn_quad    = cross_validation(model=mm.quad_dis(),         sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_gaus_nb,prc_gaus_nb,f1_gaus_nb,rec_gaus_nb,acc_gaus_nb,gmn_gaus_nb = cross_validation(model=mm.gauss_nb(),         sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_gaus_pc,prc_gaus_pc,f1_gaus_pc,rec_gaus_pc,acc_gaus_pc,gmn_gaus_pc = cross_validation(model=mm.gauss_pc(),         sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_log_reg,prc_log_reg,f1_log_reg,rec_log_reg,acc_log_reg,gmn_log_reg = cross_validation(model=mm.log_reg(),          sample_name=name, roc_area="prob", kfolds=kfolds, n_reps=n_reps)
        auc_ridge,  prc_ridge,  f1_ridge,  rec_ridge,  acc_ridge,  gmn_ridge   = cross_validation(model=mm.ridge_class(),      sample_name=name, roc_area="deci", kfolds=kfolds, n_reps=n_reps)
        auc_sgdc,   prc_sgdc,   f1_sgdc,   rec_sgdc,   acc_sgdc,   gmn_sgdc    = cross_validation(model=mm.sgdc_class(),       sample_name=name, roc_area="deci", kfolds=kfolds, n_reps=n_reps)
        auc_pass,   prc_pass,   f1_pass,   rec_pass,   acc_pass,   gmn_pass    = cross_validation(model=mm.pass_agre(),        sample_name=name, roc_area="deci", kfolds=kfolds, n_reps=n_reps)

        
    mean_auc  =  np.array([np.mean(auc_svm_ab), np.mean(auc_svm_abd), np.mean(auc_svm),     np.mean(auc_svm_lin), np.mean(auc_svm_bdt), np.mean(auc_svm_bag), np.mean(auc_ran),
                           np.mean(auc_bdt),    np.mean(auc_bag),     np.mean(auc_grad),    np.mean(auc_neur),    np.mean(auc_knn),     np.mean(auc_lin_dc),  np.mean(auc_quad),
                           np.mean(auc_gaus_nb),np.mean(auc_gaus_pc), np.mean(auc_log_reg), np.mean(auc_ridge),   np.mean(auc_sgdc),    np.mean(auc_pass)])

    mean_prc  =  np.array([np.mean(prc_svm_ab), np.mean(prc_svm_abd), np.mean(prc_svm),     np.mean(prc_svm_lin), np.mean(prc_svm_bdt), np.mean(prc_svm_bag), np.mean(prc_ran),
                           np.mean(prc_bdt),    np.mean(prc_bag),     np.mean(prc_grad),    np.mean(prc_neur),    np.mean(prc_knn),     np.mean(prc_lin_dc),  np.mean(prc_quad),
                           np.mean(prc_gaus_nb),np.mean(prc_gaus_pc), np.mean(prc_log_reg), np.mean(prc_ridge),   np.mean(prc_sgdc),    np.mean(prc_pass)])

    mean_f1   =  np.array([np.mean(f1_svm_ab), np.mean(f1_svm_abd), np.mean(f1_svm),     np.mean(f1_svm_lin), np.mean(f1_svm_bdt), np.mean(f1_svm_bag), np.mean(f1_ran),
                           np.mean(f1_bdt),    np.mean(f1_bag),     np.mean(f1_grad),    np.mean(f1_neur),    np.mean(f1_knn),     np.mean(f1_lin_dc),  np.mean(f1_quad),
                           np.mean(f1_gaus_nb),np.mean(f1_gaus_pc), np.mean(f1_log_reg), np.mean(f1_ridge),   np.mean(f1_sgdc),    np.mean(f1_pass)])

    mean_rec  =  np.array([np.mean(rec_svm_ab), np.mean(rec_svm_abd), np.mean(rec_svm),     np.mean(rec_svm_lin), np.mean(rec_svm_bdt), np.mean(rec_svm_bag), np.mean(rec_ran),
                           np.mean(rec_bdt),    np.mean(rec_bag),     np.mean(rec_grad),    np.mean(rec_neur),    np.mean(rec_knn),     np.mean(rec_lin_dc),  np.mean(rec_quad),
                           np.mean(rec_gaus_nb),np.mean(rec_gaus_pc), np.mean(rec_log_reg), np.mean(rec_ridge),   np.mean(rec_sgdc),    np.mean(rec_pass)])

    mean_acc  =  np.array([np.mean(acc_svm_ab), np.mean(acc_svm_abd), np.mean(acc_svm),     np.mean(acc_svm_lin), np.mean(acc_svm_bdt), np.mean(acc_svm_bag), np.mean(acc_ran),
                           np.mean(acc_bdt),    np.mean(acc_bag),     np.mean(acc_grad),    np.mean(acc_neur),    np.mean(acc_knn),     np.mean(acc_lin_dc),  np.mean(acc_quad),
                           np.mean(acc_gaus_nb),np.mean(acc_gaus_pc), np.mean(acc_log_reg), np.mean(acc_ridge),   np.mean(acc_sgdc),    np.mean(acc_pass)])

    mean_gmn  =  np.array([np.mean(gmn_svm_ab), np.mean(gmn_svm_abd), np.mean(gmn_svm),     np.mean(gmn_svm_lin), np.mean(gmn_svm_bdt), np.mean(gmn_svm_bag), np.mean(gmn_ran),
                           np.mean(gmn_bdt),    np.mean(gmn_bag),     np.mean(gmn_grad),    np.mean(gmn_neur),    np.mean(gmn_knn),     np.mean(gmn_lin_dc),  np.mean(gmn_quad),
                           np.mean(gmn_gaus_nb),np.mean(gmn_gaus_pc), np.mean(gmn_log_reg), np.mean(gmn_ridge),   np.mean(gmn_sgdc),    np.mean(gmn_pass)])

    std_auc  =  np.array([np.std(auc_svm_ab), np.std(auc_svm_abd), np.std(auc_svm),     np.std(auc_svm_lin), np.std(auc_svm_bdt), np.std(auc_svm_bag), np.std(auc_ran),
                          np.std(auc_bdt),    np.std(auc_bag),     np.std(auc_grad),    np.std(auc_neur),    np.std(auc_knn),     np.std(auc_lin_dc),  np.std(auc_quad),
                          np.std(auc_gaus_nb),np.std(auc_gaus_pc), np.std(auc_log_reg), np.std(auc_ridge),   np.std(auc_sgdc),    np.std(auc_pass)])
    
    std_prc  =  np.array([np.std(prc_svm_ab), np.std(prc_svm_abd), np.std(prc_svm),     np.std(prc_svm_lin), np.std(prc_svm_bdt), np.std(prc_svm_bag), np.std(prc_ran),
                          np.std(prc_bdt),    np.std(prc_bag),     np.std(prc_grad),    np.std(prc_neur),    np.std(prc_knn),     np.std(prc_lin_dc),  np.std(prc_quad),
                          np.std(prc_gaus_nb),np.std(prc_gaus_pc), np.std(prc_log_reg), np.std(prc_ridge),   np.std(prc_sgdc),    np.std(prc_pass)])
    
    std_f1   =  np.array([np.std(f1_svm_ab), np.std(f1_svm_abd), np.std(f1_svm),     np.std(f1_svm_lin), np.std(f1_svm_bdt), np.std(f1_svm_bag), np.std(f1_ran),
                          np.std(f1_bdt),    np.std(f1_bag),     np.std(f1_grad),    np.std(f1_neur),    np.std(f1_knn),     np.std(f1_lin_dc),  np.std(f1_quad),
                          np.std(f1_gaus_nb),np.std(f1_gaus_pc), np.std(f1_log_reg), np.std(f1_ridge),   np.std(f1_sgdc),    np.std(f1_pass)])
    
    std_rec  =  np.array([np.std(rec_svm_ab), np.std(rec_svm_abd), np.std(rec_svm),     np.std(rec_svm_lin), np.std(rec_svm_bdt), np.std(rec_svm_bag), np.std(rec_ran),
                          np.std(rec_bdt),    np.std(rec_bag),     np.std(rec_grad),    np.std(rec_neur),    np.std(rec_knn),     np.std(rec_lin_dc),  np.std(rec_quad),
                          np.std(rec_gaus_nb),np.std(rec_gaus_pc), np.std(rec_log_reg), np.std(rec_ridge),   np.std(rec_sgdc),    np.std(rec_pass)])

    std_acc  =  np.array([np.std(acc_svm_ab), np.std(acc_svm_abd), np.std(acc_svm),     np.std(acc_svm_lin), np.std(acc_svm_bdt), np.std(acc_svm_bag), np.std(acc_ran),
                          np.std(acc_bdt),    np.std(acc_bag),     np.std(acc_grad),    np.std(acc_neur),    np.std(acc_knn),     np.std(acc_lin_dc),  np.std(acc_quad),
                          np.std(acc_gaus_nb),np.std(acc_gaus_pc), np.std(acc_log_reg), np.std(acc_ridge),   np.std(acc_sgdc),    np.std(acc_pass)])

    std_gmn  =  np.array([np.std(gmn_svm_ab), np.std(gmn_svm_abd), np.std(gmn_svm),     np.std(gmn_svm_lin), np.std(gmn_svm_bdt), np.std(gmn_svm_bag), np.std(gmn_ran),
                          np.std(gmn_bdt),    np.std(gmn_bag),     np.std(gmn_grad),    np.std(gmn_neur),    np.std(gmn_knn),     np.std(gmn_lin_dc),  np.std(gmn_quad),
                          np.std(gmn_gaus_nb),np.std(gmn_gaus_pc), np.std(gmn_log_reg), np.std(gmn_ridge),   np.std(gmn_sgdc),    np.std(gmn_pass)])

    # tukey tests
    tukey_auc  =  tukey_test(auc_svm_ab, auc_svm_abd, auc_svm,     auc_svm_lin, auc_svm_bdt, auc_svm_bag, auc_ran,
                          auc_bdt,    auc_bag,     auc_grad,    auc_neur,    auc_knn,     auc_lin_dc,  auc_quad,
                          auc_gaus_nb,auc_gaus_pc, auc_log_reg, auc_ridge,   auc_sgdc,    auc_pass)

    tukey_prc  =  tukey_test(prc_svm_ab, prc_svm_abd, prc_svm,     prc_svm_lin, prc_svm_bdt, prc_svm_bag, prc_ran,
                             prc_bdt,    prc_bag,     prc_grad,    prc_neur,    prc_knn,     prc_lin_dc,  prc_quad,
                             prc_gaus_nb,prc_gaus_pc, prc_log_reg, prc_ridge,   prc_sgdc,    prc_pass)
    
    tukey_f1   =  tukey_test(f1_svm_ab, f1_svm_abd, f1_svm,     f1_svm_lin, f1_svm_bdt, f1_svm_bag, f1_ran,
                             f1_bdt,    f1_bag,     f1_grad,    f1_neur,    f1_knn,     f1_lin_dc,  f1_quad,
                             f1_gaus_nb,f1_gaus_pc, f1_log_reg, f1_ridge,   f1_sgdc,    f1_pass)
    
    tukey_rec  =  tukey_test(rec_svm_ab, rec_svm_abd, rec_svm,     rec_svm_lin, rec_svm_bdt, rec_svm_bag, rec_ran,
                             rec_bdt,    rec_bag,     rec_grad,    rec_neur,    rec_knn,     rec_lin_dc,  rec_quad,
                             rec_gaus_nb,rec_gaus_pc, rec_log_reg, rec_ridge,   rec_sgdc,    rec_pass)
    
    tukey_acc  =  tukey_test(acc_svm_ab, acc_svm_abd, acc_svm,     acc_svm_lin, acc_svm_bdt, acc_svm_bag, acc_ran,
                             acc_bdt,    acc_bag,     acc_grad,    acc_neur,    acc_knn,     acc_lin_dc,  acc_quad,
                             acc_gaus_nb,acc_gaus_pc, acc_log_reg, acc_ridge,   acc_sgdc,    acc_pass)
    
    tukey_gmn  =  tukey_test(gmn_svm_ab, gmn_svm_abd, gmn_svm,     gmn_svm_lin, gmn_svm_bdt, gmn_svm_bag, gmn_ran,
                             gmn_bdt,    gmn_bag,     gmn_grad,    gmn_neur,    gmn_knn,     gmn_lin_dc,  gmn_quad,
                             gmn_gaus_nb,gmn_gaus_pc, gmn_log_reg, gmn_ridge,   gmn_sgdc,    gmn_pass)
    
    # latex tables
    f_tukey_noDiv = open('./tables/tukey_'+name+'_'+boot_kfold+'_noDiv.tex', "w")
    dv.latex_table_tukey(False, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1,  tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc,  tukey_acc, mean_gmn, std_gmn,  tukey_gmn,  f_tukey_noDiv)
    f_tukey_noDiv.close()

    f_tukey_div = open('./tables/tukey_'+name+'_'+boot_kfold+'_div.tex', "w")
    dv.latex_table_tukey(True, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1, tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc, tukey_acc, mean_gmn, std_gmn, tukey_gmn, f_tukey_div)
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
