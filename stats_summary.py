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
from scipy.stats import norm, normaltest, shapiro, chisquare, kstest
import pandas as pd
from sklearn.model_selection import RepeatedKFold

# statsmodel includes
from statsmodels.stats.multicomp import pairwise_tukeyhsd,MultiComparison
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.contingency_tables import mcnemar

# framework includes
from data_preparation import data_preparation
import data_utils as du
import model_maker as mm
import data_visualization as dv
from genetic_selection import genetic_selection


def _check_A(A):
    # Validate assumptions about format of input data. Expecting response variable to be formatted as ±1
    #assert set(A) == {-1, 1}
    # If input data already is numpy array, do nothing
    if type(A) == type(np.array([])): return A
    else:
        A = A.values # convert pandas into numpy arrays
        return A


def bootstrap(sample_name, model, roc_area, selection, GA_mut=0.3, GA_score='', GA_selec='', GA_coef=0.5, n_cycles=1, train_test=False, split_frac=0.6):
    
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
            
            if selection == 'trad':
                # test data are the complement of full input data that is not considered for training
                sampled_train_no_dup = sampled_data_train.drop_duplicates(keep=False)
                sampled_data_test    = pd.concat([sample_df, sampled_train_no_dup]).drop_duplicates(keep=False)
                
                X_train, Y_train = data.dataset(sample_name=sample_name, data_set=sampled_data_train,
                                                sampling=True, split_sample=0.0, train_test=train_test)
                X_test, Y_test   = data.dataset(sample_name=sample_name, data_set=sampled_data_test,
                                                sampling=True, split_sample=0.0, train_test=train_test)                        
            elif selection == 'gene': # genetic selection
                train_indexes = sampled_data_train.index
                X,Y = data.dataset(sample_name=sample_name, data_set = sample_df, sampling = True)
                X_train, X_test, Y_train, Y_test = data.indexes_split(X, Y, train_indexes)
                print(len(X_train.index),  len(Y_train.index), 'X_train, Y_train sizes')

                GA_selection = genetic_selection(model, roc_area, X_train, Y_train, X_test, Y_test,
                                                 pop_size=10, chrom_len=100, n_gen=50, coef=GA_coef, mut_rate=GA_mut, score_type=GA_score, selec_type=GA_selec)
                GA_selection.execute()
                GA_train_indexes = GA_selection.best_population()
                X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, train_test=False, indexes=GA_train_indexes)
            
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


def mcnemar_table(y_pred1, y_pred2, y_test):
    # obtain contingency matrix for binary variables (model class predictions)
    # moreover gets whether a corrected mcnemar test is appropiate or not

    y_pred1,y_pred2,y_test=_check_A(y_pred1),_check_A(y_pred2),_check_A(y_test)
    
    # c-> correct, i->incorrect, e.g. c1==model1 prediction is correct    
    c1_c2, i1_i2, c1_i2, i1_c2 = 0,0,0,0
    for i in range(len(y_test)):
        if(y_pred1[i] == y_test[i] and y_pred2[i] == y_test[i]): c1_c2+=1#yes_yes+=1
        if(y_pred1[i] != y_test[i] and y_pred2[i] != y_test[i]): i1_i2+=1#no_no+=1
        if(y_pred1[i] == y_test[i] and y_pred2[i] != y_test[i]): c1_i2+=1#yes_no+=1
        if(y_pred1[i] != y_test[i] and y_pred2[i] == y_test[i]): i1_c2+=1#no_yes+=1
    
    matrix = np.array([[c1_c2, c1_i2],
                       [i1_c2, i1_i2]])

    corrected = True
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):            
            if int(matrix[i][j]) < 25: # following mastery
                corrected = False
    
    return matrix,corrected


def mcnemar_test(sample_name, selection='gene', model='no_div', train_test=False, GA_score='', GA_selec=''):

    if   model =='diverse': model1 = mm.adaboost_svm(True)
    elif model =='no_div' : model1 = mm.adaboost_svm(False)
    
    # fetch data
    data = data_preparation()
    X_train, Y_train, X_test, Y_test = \
    data.dataset(sample_name=sample_name,
                 sampling=False,split_sample=0.4,train_test=train_test)
    if selection == 'gene':
        GA_selection = genetic_selection(model1, "absv", X_train, Y_train, X_test, Y_test,
                                         pop_size=10, chrom_len=100, n_gen=50, coef=0.5, mut_rate=0.3, score_type=GA_score, selec_type=GA_selec)
        GA_selection.execute()
        GA_train_indexes = GA_selection.best_population()
        X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, train_test=train_test, indexes=GA_train_indexes)
    
    # train the model we are analyzing
    model1.fit(X_train, Y_train)
    y_pred1 = model1.predict(X_test)
    prec1 = precision_score(Y_test, y_pred1)
    f1_1 = f1_score(Y_test, y_pred1)
    recall1 = recall_score(Y_test, y_pred1)
    acc1 = accuracy_score(Y_test, y_pred1)
    gmean1 = np.sqrt(prec1*recall1)
    y_thresholds = model1.decision_thresholds(X_test, glob_dec=True)
    TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
    area1 = auc(FPR,TPR)
    model1.clean()

    p_values,stats,rejects,areas2,precs2,f1_s2,recalls2,accs2,gmeans2 = ([]),([]),([]),([]),([]),([]),([]),([]),([])
    names = []
    # call and train the models to compare, including the AUC calculation method
    model_auc2 = mm.model_loader(model, sample_name)
    for i in range(len(model_auc2)):
        model_auc2[i][0].fit(X_train, Y_train)
        y_pred2 = model_auc2[i][0].predict(X_test)
        prec2 = precision_score(Y_test, y_pred2)
        f1_2 = f1_score(Y_test, y_pred2)
        recall2 = recall_score(Y_test, y_pred2)
        acc2 = accuracy_score(Y_test, y_pred2)
        gmean2 = np.sqrt(prec2*recall2)

        if model_auc2[i][1]=="absv":
            y_thresholds_2 = model_auc2[i][0].decision_thresholds(X_test, glob_dec=True)
            TPR_2, FPR_2 = du.roc_curve_adaboost(y_thresholds, Y_test)
            area2 = auc(FPR_2,TPR_2)        
        elif model_auc2[i][1]=="prob":
            Y_pred_prob = model_auc2[i][0].predict_proba(X_test)[:,1]            
            area2 = roc_auc_score(Y_test, Y_pred_prob)
        elif model_auc2[i][1]=="deci":
            Y_pred_dec = model_auc2[i][0].decision_function(X_test)
            area2 = roc_auc_score(Y_test, Y_pred_dec)
        
        contingency,corrected = mcnemar_table(y_pred1, y_pred2, Y_test)        
        
        if corrected: result = mcnemar(contingency, exact=False, correction=True)
        else:         result = mcnemar(contingency, exact=True)
    
        print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        alpha = 0.05
        if result.pvalue > alpha:
            reject_null = False            
            print('Same proportions of errors, fail to reject H0')
        else:
            reject_null = True
            print('Different proportions of errors, reject H0')
            
        p_values = np.append(p_values, result.pvalue)
        stats    = np.append(stats,    result.statistic)
        rejects  = np.append(rejects,  reject_null)
        areas2   = np.append(areas2,   area2)
        precs2   = np.append(precs2,   prec2)
        f1_s2    = np.append(f1_s2,    f1_2)
        recalls2 = np.append(recalls2, recall2)
        accs2    = np.append(accs2,    acc2)
        gmeans2  = np.append(gmeans2,  gmean2)

        names.append(model_auc2[i][2])

    f_mcnemar = open('./tables/mcnemar_'+sample_name+'_'+model+'.tex', "w")
    
    dv.latex_table_mcnemar(names,p_values,stats,rejects, areas2,precs2,f1_s2,recalls2,accs2,gmeans2,
                                                       area1, prec1, f1_1, recall1, acc1, gmean1, f_mcnemar)

                
def cross_validation(sample_name, model, roc_area, selection, GA_mut=0.3, GA_score='', GA_selec='', GA_coef=0.5, kfolds=1, n_reps=1, train_test=False):
    
    # fetch data
    data = data_preparation()
    if not train_test: sample_df = data.fetch_data(sample_name)
    else: sample_train_df, sample_test_df = data.fetch_data(sample_name)
    X,Y = data.dataset(sample_name=sample_name, data_set=sample_df,
                       sampling=True, split_sample=0.0, train_test=train_test)
    
    area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores = ([]),([]),([]),([]),([]),([])
    
    # n-k fold cross validation, n_cycles = n_splits * n_repeats
    rkf = RepeatedKFold(n_splits = kfolds, n_repeats = n_reps, random_state = None)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]

        if selection == 'gene': # genetic selection
            GA_selection = genetic_selection(model, roc_area, X_train, Y_train, X_test, Y_test,
                                             pop_size=10, chrom_len=100, n_gen=50, coef=0.5, mut_rate=0.3, score_type=GA_score, selec_type=GA_selec)
            GA_selection.execute()
            GA_train_indexes = GA_selection.best_population()
            X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, train_test=train_test, indexes=GA_train_indexes)
                    
        model.fit(X_train, Y_train)                
        y_pred = model.predict(X_test)
        prec = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        acc = accuracy_score(Y_test, y_pred)
        gmean = np.sqrt(prec*recall)
        
        # calcualate roc-auc depending on the classifier
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

        # store scores
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

    
def tukey_test(score_array):

    # create INTEGER indexes to label scores
    index_array = []
    for i in range(len(score_array)):
        index_dummy = np.array([(int(i+1)) for j in range(len(score_array[i]))])
        index_array.append(index_dummy)

    # transform arrays to tuples
    score_tuple = tuple(map(tuple, score_array))
    index_tuple = tuple(map(tuple, np.array(index_array)))

    # format data for tukey function
    indexes= np.concatenate(index_tuple, axis=0)                            
    values = np.concatenate(score_tuple, axis=0)                            
    data   = {'means':values, 'group':indexes}

    # perform the pairwise tukey test
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
    # arrays to store the scores
    mean_auc,mean_prc,mean_f1,mean_rec,mean_acc,mean_gmn = ([]),([]),([]),([]),([]),([])
    std_auc,std_prc,std_f1,std_rec,std_acc,std_gmn = ([]),([]),([]),([]),([]),([])
    auc_values,prc_values,f1_values,rec_values,acc_values,gmn_values = [],[],[],[],[],[]
    names = []

    # load models and auc methods
    models_auc = mm.model_loader("boot", name)
    
    for i in range(len(models_auc)):
        if boot_kfold == 'bootstrap':
            auc, prc, f1, rec, acc, gmn = bootstrap(sample_name=name, model=models_auc[i][1], roc_area=models_auc[i][2],
                                                    selection=models_auc[i][3], GA_mut=models_auc[i][4], GA_score=models_auc[i][5],
                                                    GA_selec=models_auc[i][6], GA_coef=models_auc[i][7], n_cycles=n_cycles)
        elif boot_kfold == 'kfold':
            auc, prc, f1, rec, acc, gmn = cross_validation(sample_name=name, model=models_auc[i][1],  roc_area=models_auc[i][2],
                                                           selection=models_auc[i][3], GA_mut=models_auc[i][4], GA_score=models_auc[i][5],
                                                           GA_selec=models_auc[i][6], GA_coef=models_auc[i][7], kfolds=kfolds, n_reps=n_reps)
        auc_values.append(auc)
        prc_values.append(prc)
        f1_values.append(f1)
        rec_values.append(rec)
        acc_values.append(acc)
        gmn_values.append(gmn)
        
        mean_auc = np.append(mean_auc,  np.mean(auc))
        mean_prc = np.append(mean_prc,  np.mean(prc))
        mean_f1  = np.append(mean_f1,   np.mean(f1))
        mean_rec = np.append(mean_rec,  np.mean(rec))
        mean_acc = np.append(mean_acc,  np.mean(acc))
        mean_gmn = np.append(mean_gmn,  np.mean(gmn))
        
        std_auc = np.append(std_auc,  np.std(auc))
        std_prc = np.append(std_prc,  np.std(prc))
        std_f1  = np.append(std_f1,   np.std(f1))
        std_rec = np.append(std_rec,  np.std(rec))
        std_acc = np.append(std_acc,  np.std(acc))
        std_gmn = np.append(std_gmn,  np.std(gmn))
        
        # store model names, for later use in latex tables
        names.append(models_auc[i][0])
    
    # tukey tests
    tukey_auc  =  tukey_test(np.array(auc_values))
    tukey_prc  =  tukey_test(np.array(prc_values))
    tukey_f1   =  tukey_test(np.array(f1_values))  
    tukey_rec  =  tukey_test(np.array(rec_values)) 
    tukey_acc  =  tukey_test(np.array(acc_values))
    tukey_gmn  =  tukey_test(np.array(gmn_values))
                                 
    # latex tables
    f_tukey_noDiv = open('./tables/tukey_'+name+'_'+boot_kfold+'_noDiv.tex', "w")
    dv.latex_table_tukey(names, False, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1,  tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc,  tukey_acc, mean_gmn, std_gmn,  tukey_gmn,  f_tukey_noDiv)
    f_tukey_noDiv.close()

    f_tukey_div = open('./tables/tukey_'+name+'_'+boot_kfold+'_div.tex', "w")
    dv.latex_table_tukey(names, True, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1, tukey_f1,
                         mean_rec, std_rec,  tukey_rec, mean_acc, std_acc, tukey_acc, mean_gmn, std_gmn, tukey_gmn, f_tukey_div)
    f_tukey_div.close()

    # p_gaus, alpha = ss.normal_test(sample=auc_svm_boost,     alpha=0.05, verbose=True)
    
    # ss.anova_test(auc_svm_boost,     
    #               auc_svm_boost_div, 
    #               auc_svm_single,    
    #               auc_ran_forest,    
    #               auc_bdt_forest,    
    #               auc_net_neural,    
    #               auc_knn_neighb
