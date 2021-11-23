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
ensem_single = str(sys.argv[5])# use ensemble or standard classifiers

model_auc = mm.model_loader_batch(process, ensemble_single=ensem_single)[1]
model_auc_names = mm.model_loader_batch(process, ensemble_single=ensem_single)[0]
n_cycles = 10
k_folds  = 50
n_reps   = 1

print('sample:', name, 'model name:', model_auc[0], '  validation', boot_kfold)
start = datetime.datetime.now()
if(boot_kfold=="boot"):
    auc, prc, f1, rec, acc, gmn, time, n_class, n_train = ss.bootstrap(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                                                       selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                                                       GA_selec=model_auc[6], GA_coef=model_auc[7], n_cycles=n_cycles, path=path)
elif(boot_kfold=="kfold"):
    auc, prc, f1, rec, acc, gmn, time, n_class, n_train = ss.cross_validation(sample_name=name, model=model_auc[1], roc_area=model_auc[2],
                                                                              selection=model_auc[3], GA_mut=model_auc[4], GA_score=model_auc[5],
                                                                              GA_selec=model_auc[6], GA_coef=model_auc[7], kfolds=k_folds, n_reps=n_reps, path=path)
    
col_auc = pd.DataFrame(data=auc,    columns=["auc"])
col_prc = pd.DataFrame(data=prc,    columns=["prc"])
col_f1  = pd.DataFrame(data=f1,     columns=["f1"])
col_rec = pd.DataFrame(data=rec,    columns=["rec"])
col_acc = pd.DataFrame(data=acc,    columns=["acc"])
col_gmn = pd.DataFrame(data=gmn,    columns=["gmn"])
col_time= pd.DataFrame(data=time,   columns=["time"])
col_base= pd.DataFrame(data=n_class,columns=["n_base"])
col_size= pd.DataFrame(data=n_train,columns=["n_train"])
df = pd.concat([col_auc["auc"], col_prc["prc"], col_f1["f1"], col_rec["rec"], col_acc["acc"], col_gmn["gmn"], col_time["time"], col_base["n_base"], col_size["n_train"]],
               axis=1, keys=["auc", "prc", "f1", "rec", "acc", "gmn", "time", "n_base", "n_train"])
name_csv = path+"/stats_results_fast/"+name+"/"+boot_kfold+"/"+model_auc[0]+"_"+boot_kfold+".csv" 
df.to_csv(str(name_csv), index=False)

end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed time = " + str(elapsed_time))
print(model_auc[0], name)
print(df)
print('All names')
for i in range(len(model_auc_names)):
    print(model_auc_names[i][0])
