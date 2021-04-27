#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import sys
import pandas as pd
import stats_summary as ss
import model_maker as mm

process = int(sys.argv[1])    # batch process
name = str(sys.argv[2])       # sample name
path = str(sys.argv[3])       # path where code lives
boot_kfold = str(sys.argv[4]) # use bootstrap or kfold

model_auc = mm.model_loader_batch(process)
n_cycles=4
k_folds = 2
n_reps = 2

if(boot_kfold=="boot"):
    auc, prc, f1, rec, acc, gmn = ss.bootstrap(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                               selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                               GA_selec=model_auc[6], GA_coef=model_auc[7], n_cycles=n_cycles, path=path)
elif(boot_kfold=="kfold"):
    auc, prc, f1, rec, acc, gmn = ss.cross_validation(sample_name=name, model=model_auc[i][1], roc_area=model_auc[i][2],
                                                      selection=model_auc[i][3], GA_mut=model_auc[i][4], GA_score=model_auc[i][5],
                                                      GA_selec=model_auc[i][6], GA_coef=model_auc[i][7], kfolds=kfolds, n_reps=n_reps, path=path)

col_auc = pd.DataFrame(data=auc, columns=["auc"])
col_prc = pd.DataFrame(data=prc, columns=["prc"])
col_f1  = pd.DataFrame(data=f1,  columns=["f1"])
col_rec = pd.DataFrame(data=rec, columns=["rec"])
col_acc = pd.DataFrame(data=acc, columns=["acc"])
col_gmn = pd.DataFrame(data=gmn, columns=["gmn"])
df = pd.concat([col_auc["auc"], col_prc["prc"], col_f1["f1"], col_rec["rec"], col_acc["acc"], col_gmn["gmn"]],
               axis=1, keys=["auc", "prc", "f1", "rec", "acc", "gmn"])
name_csv = path+"/stats_results/"+name+"/"+model_auc[0]+"_"+boot_kfold+".csv" 
df.to_csv(str(name_csv), index=False)

print(model_auc[0], name)
print(df)