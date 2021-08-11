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
import datetime

process = int(sys.argv[1])     # batch process
name = str(sys.argv[2])        # sample name
path = str(sys.argv[3])        # path where code lives
boot_kfold = str(sys.argv[4])  # use bootstrap or kfold
ensem_single = str(sys.argv[5])# use ensemble or stardad classifiers

model_auc = mm.model_loader_batch(process, ensemble_single=ensem_single)[1]
model_auc_names = mm.model_loader_batch(process, ensemble_single=ensem_single)[0]
n_cycles = 10
k_folds  = 5
n_reps   = 2

print('sample:', name, 'model name:', model_auc[0], '  validation', boot_kfold)
start = datetime.datetime.now()
if(boot_kfold=="boot"):
    auc, prc, f1, rec, acc, gmn, time = ss.bootstrap(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                                     selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                                     GA_selec=model_auc[6], GA_coef=model_auc[7], n_cycles=n_cycles, path=path)
elif(boot_kfold=="kfold"):
    auc, prc, f1, rec, acc, gmn, time = ss.cross_validation(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                                            selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                                            GA_selec=model_auc[6], GA_coef=model_auc[7], kfolds=k_folds, n_reps=n_reps, path=path)

col_auc = pd.DataFrame(data=auc, columns=["auc"])
col_prc = pd.DataFrame(data=prc, columns=["prc"])
col_f1  = pd.DataFrame(data=f1,  columns=["f1"])
col_rec = pd.DataFrame(data=rec, columns=["rec"])
col_acc = pd.DataFrame(data=acc, columns=["acc"])
col_gmn = pd.DataFrame(data=gmn, columns=["gmn"])
col_time= pd.DataFrame(data=time,columns=["time"])
df = pd.concat([col_auc["auc"], col_prc["prc"], col_f1["f1"], col_rec["rec"], col_acc["acc"], col_gmn["gmn"], col_time["time"]],
               axis=1, keys=["auc", "prc", "f1", "rec", "acc", "gmn", "time"])
name_csv = path+"/stats_results/"+name+"/"+boot_kfold+"/"+model_auc[0]+"_"+boot_kfold+".csv" 
df.to_csv(str(name_csv), index=False)

end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed time = " + str(elapsed_time))
print(model_auc[0], name)
print(df)
print('All names')
for i in range(len(model_auc_names)):
    print(model_auc_names[i][0])
