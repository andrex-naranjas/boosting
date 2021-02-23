#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# visualization module

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import math as math
from sklearn.metrics import auc

# frame plots
def plot_frame(frame,name,xlabel,ylabel,yUserRange,ymin,ymax,sample):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(frame,label=sample)
    if yUserRange:
        plt.ylim(ymin,ymax)
    # plt.text(0.15, 0.9,'$\mu$={}, $\sigma$={}'.format(round(1.0,1), round(1.0,1)),
    #          ha='center', va='center', transform=ax.transAxes)
    plt.legend(frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig('./plots/'+name+'_'+sample+'.pdf')
    plt.close()

# 2d test error plot as function of sigma and c SVM parameters
def plot_2dmap(matrix,sigmin,sigmax,cmin,cmax,sample_name):

    tick_x = [math.floor(sigmin),0,math.floor(sigmax)]
    tick_y = [math.floor(cmax),math.floor(cmax/2),math.floor(cmin)]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticklabels(tick_x)
    ax.set_yticklabels(tick_y)

    # ax.set_xticks(np.arange(matrix.shape[1])) # show all ticks
    # ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks([0,matrix.shape[1]/2, matrix.shape[1]-1])
    ax.set_yticks([0,matrix.shape[0]/2, matrix.shape[0]-1])

    # loop over data dimensions and create text annotations.
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            text = ax.text(j, i, math.floor(100*matrix[i,j]),
                           ha="center", va="center", color="black")

    ax.set_title('Test Error (%) - '+sample_name+' dataset')
    fig.tight_layout()
    plt.xlabel('ln $\sigma$')
    plt.ylabel('ln C')
    plt.savefig('./plots/2dplot_'+sample_name+'.pdf')
    plt.close()

def plot_hist_frame(frame, sample_name):

    var = ['D0_m', 'D0_p', 'p0_p']
    for i in range(len(var)):

        if(var[i]=='D0_m'):
            xlow, xhigh, xlabel = 1.8,1.95,'$M(D^0)$ [GeV/$c^2$]'
        if(var[i]=='D0_p'):
            xlow, xhigh, xlabel = 0,8.0,'$p(D^0) [GeV/$c$]$'
        if(var[i]=='p0_p'):
            xlow, xhigh, xlabel = 0,4.5,'$p(\pi^0) [GeV/$c$]$'

        fig, ax = plt.subplots(figsize=(8,5))
        h = ax.hist(frame[var[i]][frame.Class==1],
                    bins=100, range=(xlow,xhigh),
                    histtype='stepfilled', lw=1,
                    label="Signal", edgecolor='black')

        h = ax.hist(frame[var[i]][frame.Class==-1],
                    bins=100, range=(xlow,xhigh),
                    histtype='step', lw=2,
                    label="Background")
        ax.legend(loc="best")
        ax.set_xlabel(xlabel, fontsize=16)
        #ax.grid()
        ax.set_xlim(xlow,xhigh)
        fig.tight_layout()

        plt.savefig('./plots/mva_'+var[i]+'_'+sample_name+'.pdf')
        plt.close()

def plot_roc_curve(TPR,FPR,sample,real,glob_local,name,kernel,nClass):

    if(real=='sorted'):
        TPR = np.sort(TPR,axis=None)
        FPR = np.sort(FPR,axis=None)

    if glob_local: glob_local='global'
    else:          glob_local='local'

    area = auc(FPR,TPR)
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f, N = %0.0f)'  %(area, nClass), linestyle="-", marker="o")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve -' + sample)
    plt.legend(loc="lower right")
    plt.savefig('./plots/roc_curve_'+sample+'_'+real+'_'+glob_local+'_'+name+'_'+kernel+'.png')
    output = pd.DataFrame({'False positive rate': FPR,'True positive rate': TPR, 'Area': area})
    output.to_csv('output/' + sample +  '/' + 'BoostSVM_ROC.csv', index=False)
    plt.close()
    

def latex_table_tukey(isDiverse, auc_val, auc_test, prc_val,  prc_test, f1_val,  f1_test,
                                 rec_val, rec_test, acc_val,  acc_test, gmn_val, gmn_test,  f_out):
                      
    print("\\begin{tabular}{c | c  c  c | c c c | c c c | c c c | c c c | c c c }\hline \hline", file=f_out)
    print("Model & $\mu_{AUC}$  & p-val  &  Reject $H_{0}$ & $\mu_{prc}$  & p-val  &  Rjct. $H_{0}$ & $\mu_{f1}$  & p-val  &  Rjct. $H_{0}$ & $\mu_{rec}$  & p-val  &  Rjct. $H_{0}$ & $\mu_{acc}$  & p-val  &  Rjct. $H_{0}$ & $\mu_{gmn}$  & p-val  &  Rjct. $H_{0}$   \\\  \hline", file=f_out)

    nAlgos = len(auc_val)-1
    off_set = nAlgos

    reject_auc = ''
    reject_prc = ''
    reject_f1  = ''
    reject_rec = ''
    reject_acc = ''
    reject_gmn = ''
    
    for i in range(nAlgos):
        k=0,0
        j = i + 1
        if isDiverse:
            if i==0:
                k = i
                j = i
            else: k = i + off_set
        else: k = i
        
            
        if auc_test.reject[k]: reject_auc = '\\checkmark'
        elif not auc_test.reject[k]: reject_auc = '\\xmark'        
        if prc_test.reject[k]: reject_prc = '\\checkmark'
        elif not prc_test.reject[k]: reject_prc = '\\xmark'
        if f1_test.reject[k]: reject_f1 = '\\checkmark'
        elif not f1_test.reject[k]: reject_f1 = '\\xmark'
        if rec_test.reject[k]: reject_rec = '\\checkmark'
        elif not rec_test.reject[k]: reject_rec = '\\xmark'
        if acc_test.reject[k]: reject_acc = '\\checkmark'
        elif not acc_test.reject[k]: reject_acc = '\\xmark'
        if gmn_test.reject[k]: reject_gmn = '\\checkmark'
        elif not gmn_test.reject[k]: reject_gmn = '\\xmark'        
        
        print('model:',i, ' & ',
              round(auc_val[j],2),' & ', round(auc_test.pvalues[k],3), ' & ', reject_auc,' & ',
              round(prc_val[j],2),' & ', round(prc_test.pvalues[k],3), ' & ', reject_prc,' & ',
              round(f1_val[j],2), ' & ', round(f1_test.pvalues[k],3),  ' & ', reject_f1, ' & ',
              round(rec_val[j],2),' & ', round(rec_test.pvalues[k],3), ' & ', reject_rec,' & ',
              round(acc_val[j],2),' & ', round(acc_test.pvalues[k],3), ' & ', reject_acc,' & ',
              round(gmn_val[j],2),' & ', round(gmn_test.pvalues[k],3), ' & ', reject_gmn,
              ' \\\ ', file=f_out)
        
                
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\caption{Tukey statistics test. Scores of current classifier  --AUC:", round(auc_val[0],2), "-- prc:", round(prc_val[0],2), "-- f1:", round(f1_val[0],2), '-- REC' ,round(rec_val[0],2), "-- acc:", round(acc_val[0],2), "-- gmn:", round(gmn_val[0],2),"}", file=f_out)
    print("\label{tab:tukey}", file=f_out)


    # print(test.pvalues, type(test.pvalues), len(test.pvalues))
    # print(test.reject, type(test.reject), len(test.reject))
    # print(test.confint, type(test.confint), len(test.confint))
    # print(test.std_pairs, type(test.std_pairs), len(test.std_pairs))





    
                
#             if not self.asymmetric or not bootstrap:
#                 print(quantum[i], round(mass,1), '$\\pm',round(error,1), '$ &', exp[i], '$\\pm', delta_exp[i], '$ &', round(decay,3), ' & $xx\pm xx$ \\\ ', file=f_paper)
#             else:
#                 if not np.isnan(up_decay):
#                     if exp[i]!=0.:
#                         print(quantum[i],'$',round(mass,1),'^{+',round(error_up,1),'}_{',round(error_dn,1),'}$',  '& $',exp[i],'\\pm',delta_exp[i], '$ & $',
#                               round(decay,3),'^{+', round(up_decay,3),'}_{', round(dn_decay,3),'}$', ' & $ xx\pm xx$ \\\ ', file=f_paper)
#                     else:
#                         print(quantum[i],'$',round(mass,1),'^{+',round(error_up,1),'}_{',round(error_dn,1),'}$',  '& $\\dagger $ & $',
#                               round(decay,3),'^{+', round(up_decay,3),'}_{', round(dn_decay,3),'}$', ' & $ xx\pm xx$ \\\ ', file=f_paper)                        
#                 else:
#                     if exp[i]!=0.:
#                         print(quantum[i],'$',round(mass,1),'^{+',round(error_up,1),'}_{',round(error_dn,1),'}$',  '& $',exp[i],'\\pm',delta_exp[i], '$ & ',
#                               '$\\dagger\\dagger$', '& $ xx\pm xx$  \\\ ', file=f_paper)
#                     else:
#                         print(quantum[i],'$',round(mass,1),'^{+',round(error_up,1),'}_{',round(error_dn,1),'}$',  '& $\\dagger$ & ',
#                               '$\\dagger\\dagger$', '& $ xx\pm xx$  \\\ ', file=f_paper)



#     def latex_header(self,table_file,paper):        
#         if not paper:
#         else:
#             print("\\begin{tabular}{c | c  c c c }\hline \hline", file=table_file)
#             print(" State     & Predicted Mass   & Experimental Mass & Predicted Width & Experimental Width   \\\ ", file=table_file)
#             print("           &      (MeV)       &    (MeV)          &      (MeV)      & $\Gamma_{tot}$ (MeV) \\\ \hline", file=table_file)
                    

#     def latex_bottom(self, table_file,diff_pred, diff_sample,paper):
#         label = 'paper'
#         if not paper:
#             print('\hline', file=table_file)
#             print("  &  &  & Total diff & ",round(diff_pred), " &" ,round(diff_sample,1),"\\\ ", file=table_file)
#             label = 'note'
            


    
    
# def decay_table_channels(baryons, channel_widths, sym_errors, f_out):

#     channel_widths = np.array(channel_widths)

#     name_decays=[]
#     name_decays.append('State')
#     for i in range(len(channel_widths[0])):
#         name_decays.append(du.latex_decay_label(baryons,i+1))
        
#     name_decays.append('Tot $\\Gamma$')                   
#     du.print_header_latex(name_decays, f_out)    
    
#     for i in range(len(channel_widths)):
#         if sym_errors is not None:
#             error = sym_errors[i]
#         else: error = 0
        
#         du.print_row_latex(state_name=str(i+1), state_decays=channel_widths[i],
#                            errors=error, f_out=f_out)
        
#     du.print_bottom_latex(baryons,f_out)
