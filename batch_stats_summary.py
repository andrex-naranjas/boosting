#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import stats_summary as ss
import model_maker as mm
import sys

process = int(sys.argv[1])

model_auc = mm.model_loader_batch(process)

n_cycles=2
name = str(sys.argv[2])

auc, prc, f1, rec, acc, gmn = ss.bootstrap(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                           selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                           GA_selec=model_auc[6], GA_coef=model_auc[7], n_cycles=n_cycles)
print(model_auc[0], name)
print(auc, prc, f1, rec, acc, gmn)

# def stats_results(name, n_cycles, kfolds, n_reps, boot_kfold ='', split_frac=0.6):
#     # arrays to store the scores
#     mean_auc,mean_prc,mean_f1,mean_rec,mean_acc,mean_gmn = ([]),([]),([]),([]),([]),([])
#     std_auc,std_prc,std_f1,std_rec,std_acc,std_gmn = ([]),([]),([]),([]),([]),([])
#     auc_values,prc_values,f1_values,rec_values,acc_values,gmn_values = [],[],[],[],[],[]
#     names = []

#     # load models and auc methods
#     models_auc = mm.model_loader("boot", name)
    
#     for i in range(len(models_auc)):
#         if boot_kfold == 'bootstrap':
#         elif boot_kfold == 'kfold':
#             auc, prc, f1, rec, acc, gmn = cross_validation(sample_name=name, model=models_auc[i][1],  roc_area=models_auc[i][2],
#                                                            selection=models_auc[i][3], GA_mut=models_auc[i][4], GA_score=models_auc[i][5],
#                                                            GA_selec=models_auc[i][6], GA_coef=models_auc[i][7], kfolds=kfolds, n_reps=n_reps)
#         auc_values.append(auc)
#         prc_values.append(prc)
#         f1_values.append(f1)
#         rec_values.append(rec)
#         acc_values.append(acc)
#         gmn_values.append(gmn)
        
#         mean_auc = np.append(mean_auc,  np.mean(auc))
#         mean_prc = np.append(mean_prc,  np.mean(prc))
#         mean_f1  = np.append(mean_f1,   np.mean(f1))
#         mean_rec = np.append(mean_rec,  np.mean(rec))
#         mean_acc = np.append(mean_acc,  np.mean(acc))
#         mean_gmn = np.append(mean_gmn,  np.mean(gmn))
        
#         std_auc = np.append(std_auc,  np.std(auc))
#         std_prc = np.append(std_prc,  np.std(prc))
#         std_f1  = np.append(std_f1,   np.std(f1))
#         std_rec = np.append(std_rec,  np.std(rec))
#         std_acc = np.append(std_acc,  np.std(acc))
#         std_gmn = np.append(std_gmn,  np.std(gmn))
        
#         # store model names, for later use in latex tables
#         names.append(models_auc[i][0])
    
#     # tukey tests
#     tukey_auc  =  tukey_test(np.array(auc_values))
#     tukey_prc  =  tukey_test(np.array(prc_values))
#     tukey_f1   =  tukey_test(np.array(f1_values))  
#     tukey_rec  =  tukey_test(np.array(rec_values)) 
#     tukey_acc  =  tukey_test(np.array(acc_values))
#     tukey_gmn  =  tukey_test(np.array(gmn_values))
                                 
#     # latex tables
#     f_tukey_noDiv = open('./tables/tukey_'+name+'_'+boot_kfold+'_noDiv.tex', "w")
#     dv.latex_table_tukey(names, False, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1,  tukey_f1,
#                          mean_rec, std_rec,  tukey_rec, mean_acc, std_acc,  tukey_acc, mean_gmn, std_gmn,  tukey_gmn,  f_tukey_noDiv)
#     f_tukey_noDiv.close()

#     f_tukey_div = open('./tables/tukey_'+name+'_'+boot_kfold+'_div.tex', "w")
#     dv.latex_table_tukey(names, True, mean_auc, std_auc, tukey_auc, mean_prc, std_prc,  tukey_prc, mean_f1, std_f1, tukey_f1,
#                          mean_rec, std_rec,  tukey_rec, mean_acc, std_acc, tukey_acc, mean_gmn, std_gmn, tukey_gmn, f_tukey_div)
#     f_tukey_div.close()





