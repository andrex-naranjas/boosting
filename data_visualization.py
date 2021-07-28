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


# simple plots
def simple_plot(sample,name='AUC',xlabel='metric', pval=0, alpha_in=0.05):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(sample, 20, density=True, label = 'Sampling')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, np.mean(sample), np.std(sample))
    p, alpha = pval, alpha_in
    plt.plot(x,y,label='Gaussian fit')
    plt.text(0.15, 0.9,'$\\mu$={}, $\\sigma$={}'.format(round(np.mean(sample),1), round(np.std(sample),4)),
             ha='center', va='center', transform=ax.transAxes)
    plt.text(0.15, 0.8,'$p_{{val}}$={}, $\\alpha$={}'.format(round(p,3), alpha),
             ha='center', va='center', transform=ax.transAxes)
    plt.legend(frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel('Arbitrary Units')
    #plt.title(quark+' mesons')
    plt.savefig('./plots/'+name+'.pdf')
    plt.close()
    
# 2d test error plot as function of sigma and c SVM parameters
def plot_2dmap(matrix,sigmin,sigmax,cmin,cmax,sample_name, my_kernel='rbf'):

    if my_kernel == 'rbf':
        sigmax,sigmin=100,0.00
        cmax,cmin=100,0
    elif my_kernel == 'sigmoid':
        sigmax,sigmin=0.1,0.00
        cmax,cmin=100,0
    elif my_kernel == 'poly' or my_kernel == 'linear':
        sigmax,sigmin=0.1,0.00
        cmax,cmin=10,0

    # tick_x = [math.floor(sigmax), math.floor(sigmax/2) , math.floor(sigmin)]
    # tick_y = [math.floor(cmax),math.floor(cmax/2),math.floor(cmin)]
    half_sig = sigmax/2
    if my_kernel =='rbf': half_sig = int(half_sig)
    
    tick_x = [sigmax, half_sig, int(sigmin)]
    tick_y = [cmax,   int(cmax/2),   cmin]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    
    ax.set_yticklabels(tick_y)
    ax.yaxis.label.set_size(15)
    
    ax.set_xticklabels(tick_x)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.xaxis.label.set_size(15)
    ax.set_xlabel('$\gamma$')
    
    #ax.spines['left'].set_visible(False)

    # ax.set_xticks(np.arange(matrix.shape[1])) # show all ticks
    # ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks([0,matrix.shape[1]/2, matrix.shape[1]-1])
    ax.set_yticks([0,matrix.shape[0]/2, matrix.shape[0]-1])

    # loop over data dimensions and create text annotations.
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            text = ax.text(j, i, math.floor(100*matrix[i,j]),
                           ha="center", va="center", color="black")

    #ax.set_title('Test Error (%) - '+sample_name+' dataset')
    fig.tight_layout()
    #plt.xlabel('$\gamma$')
    plt.ylabel('C')
    plt.savefig('./plots/2dplot_'+sample_name+'.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close()


def plot_ordered_stats_summary(values, names):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    size_x = len(values)
    

    x = np.linspace(0, size_x, size_x)
    y = values
    ax.set_xticklabels(names, rotation=45, ha='right')    
    plt.scatter(x,y,label='random label')
    
    plt.rcParams["figure.figsize"] = (20,3)
    
    # plt.text(0.15, 0.9,'$\\mu$={}, $\\sigma$={}'.format(round(np.mean(sample),1), round(np.std(sample),4)),
    #          ha='center', va='center', transform=ax.transAxes)
    # plt.text(0.15, 0.8,'$p_{{val}}$={}, $\\alpha$={}'.format(round(p,3), alpha),
    #          ha='center', va='center', transform=ax.transAxes)
    
    #plt.xlabel(xlabel)
    plt.ylabel('Arbitrary Units')
    #plt.title(quark+' mesons')
    plt.savefig('./gordito.pdf')
    plt.close()


    
    
def plot_stats_2d(matrix, sample_name):

    # tick_x = [math.floor(sigmin),0,math.floor(sigmax)]
    # tick_y = [math.floor(cmax),math.floor(cmax/2),math.floor(cmin)]

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='seismic')

    # ax.set_xticklabels(tick_x)
    # ax.set_yticklabels(tick_y)

    ax.set_xticks(np.arange(matrix.shape[1])) # show all ticks
    ax.set_yticks(np.arange(matrix.shape[0]))
    # ax.set_xticks([0,matrix.shape[1]/2, matrix.shape[1]-1]) # show some ticks
    # ax.set_yticks([0,matrix.shape[0]/2, matrix.shape[0]-1])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # loop over data dimensions and create text annotations.
    # for i in range(matrix.shape[1]):
    #     for j in range(matrix.shape[0]):
    #         text = ax.text(i, j, math.floor(1*matrix[i,j]),
    #                        ha="center", va="center", color="black")

    ax.set_title('pvalue test - '+sample_name+' dataset', color=(0.1, 0, 0.5))
    fig.tight_layout()
    # plt.xlabel('Classifier')
    # plt.ylabel('Classifier')
    plt.savefig('./plots/2dplot_stats_'+sample_name+'.pdf')
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
    

def latex_table_tukey(names, sample, auc_val, auc_error, auc_test, prc_val, prc_error,  prc_test, f1_val,  f1_error,  f1_test,
                      rec_val, rec_error, rec_test, acc_val, acc_error,  acc_test, gmn_val, gmn_error, gmn_test,  f_out):
                      
    print("\\begin{tabular}{c | c  c  c | c c c | c c c | c c c | c c c | c c c }\hline \hline", file=f_out)
    print("Model & $\mu_{AUC}$  & p-val  &  R.$H_{0}$ & $\mu_{prc}$  & p-val  &  R.$H_{0}$ & $\mu_{f1}$  & p-val  &  R.$H_{0}$ & $\mu_{rec}$  & p-val  &  R.$H_{0}$ & $\mu_{acc}$  & p-val  &  R.$H_{0}$ & $\mu_{gmn}$  & p-val  &  R.$H_{0}$   \\\  \hline", file=f_out)

    reject_auc = ''
    reject_prc = ''
    reject_f1  = ''
    reject_rec = ''
    reject_acc = ''
    reject_gmn = ''

    for k in range(len(auc_val)-1): # loop number of algorithms
        j = k + 1                    
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
        
        print(names[j], '&',
              '$', round(auc_val[j],2), '\\pm',round(auc_error[j],2), "$",'&', round(auc_test.pvalues[k],3), '&', reject_auc,' & ',
              '$', round(prc_val[j],2), '\\pm',round(prc_error[j],2), "$",'&', round(prc_test.pvalues[k],3), '&', reject_prc,' & ',
              '$', round(f1_val[j],2),  '\\pm',round( f1_error[j],2), "$",'&', round(f1_test.pvalues[k],3),  '&', reject_f1, ' & ',
              '$', round(rec_val[j],2), '\\pm',round(rec_error[j],2), "$",'&', round(rec_test.pvalues[k],3), '&', reject_rec,' & ',
              '$', round(acc_val[j],2), '\\pm',round(acc_error[j],2), "$",'&', round(acc_test.pvalues[k],3), '&', reject_acc,' & ',
              '$', round(gmn_val[j],2), '\\pm',round(gmn_error[j],2), "$",'&', round(gmn_test.pvalues[k],3), '&', reject_gmn,
              ' \\\ ', file=f_out)
                        
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\caption{Tukey statistics test. Scores of \\textbf{",sample, names[0],"} --AUC:",round(auc_val[0], 3),'$\\pm',round(auc_error[0],3), "$",
                                                                        "--prc:",round(prc_val[0], 3),'$\\pm',round(prc_error[0],3), "$",
                                                                        "--f1s:",round( f1_val[0], 3),'$\\pm',round( f1_error[0],3), "$",
                                                                        "--REC:",round(rec_val[0], 3),'$\\pm',round(rec_error[0],3), "$",
                                                                       "-- acc:",round(acc_val[0], 3),'$\\pm',round(acc_error[0],3), "$",
                                                                        "--gmn:",round(gmn_val[0], 3),'$\\pm',round(gmn_error[0],3), "$","}", file=f_out)
    print("\label{tab:tukey}", file=f_out)
    

def latex_table_student(names, sample, auc_val, auc_error, auc_pval, prc_val, prc_error,  prc_pval, f1_val,  f1_error,  f1_pval,
                        rec_val, rec_error, rec_pval, acc_val, acc_error,  acc_pval, gmn_val, gmn_error, gmn_pval, f_out):
    
    print("\\begin{tabular}{c | c  c  c | c c c | c c c | c c c | c c c | c c c }\hline \hline", file=f_out)
    print("Model & $\mu_{AUC}$  & p-val  &  R.$H_{0}$ & $\mu_{prc}$  & p-val  &  R.$H_{0}$ & $\mu_{f1}$  & p-val  &  R.$H_{0}$ & $\mu_{rec}$  & p-val  &  R.$H_{0}$ & $\mu_{acc}$  & p-val  &  R.$H_{0}$ & $\mu_{gmn}$  & p-val  &  R.$H_{0}$   \\\  \hline", file=f_out)

    reject_auc = ''
    reject_prc = ''
    reject_f1  = ''
    reject_rec = ''
    reject_acc = ''
    reject_gmn = ''
    alpha = 0.05

    for k in range(len(auc_val)-1): # loop number of algorithms
        j = k + 1                    
        if auc_pval[j] < alpha : reject_auc = '\\checkmark'
        else:                    reject_auc = '\\xmark'        
        if prc_pval[j] < alpha : reject_prc = '\\checkmark'
        else:                    reject_prc = '\\xmark'
        if f1_pval[j]  < alpha : reject_f1  = '\\checkmark'
        else:                    reject_f1  = '\\xmark'
        if rec_pval[j] < alpha:  reject_rec = '\\checkmark'
        else:                    reject_rec = '\\xmark'
        if acc_pval[j] < alpha : reject_acc = '\\checkmark'
        else:                    reject_acc = '\\xmark'
        if gmn_pval[j] < alpha:  reject_gmn = '\\checkmark'
        else:                    reject_gmn = '\\xmark'        
        
        print(names[j], '&',
              '$', round(auc_val[j],2), '\\pm',round(auc_error[j],2), "$",'&', round(auc_pval[j],3), '&', reject_auc,' & ',
              '$', round(prc_val[j],2), '\\pm',round(prc_error[j],2), "$",'&', round(prc_pval[j],3), '&', reject_prc,' & ',
              '$', round(f1_val[j],2),  '\\pm',round( f1_error[j],2), "$",'&', round( f1_pval[j],3), '&', reject_f1, ' & ',
              '$', round(rec_val[j],2), '\\pm',round(rec_error[j],2), "$",'&', round(rec_pval[j],3), '&', reject_rec,' & ',
              '$', round(acc_val[j],2), '\\pm',round(acc_error[j],2), "$",'&', round(acc_pval[j],3), '&', reject_acc,' & ',
              '$', round(gmn_val[j],2), '\\pm',round(gmn_error[j],2), "$",'&', round(gmn_pval[j],3), '&', reject_gmn,
              ' \\\ ', file=f_out)
                        
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\caption{t-student statistics test. Scores of \\textbf{",sample, names[0],"} --AUC:",round(auc_val[0], 3),'$\\pm',round(auc_error[0],3), "$",
                                                                        "--prc:",round(prc_val[0], 3),'$\\pm',round(prc_error[0],3), "$",
                                                                        "--f1s:",round( f1_val[0], 3),'$\\pm',round( f1_error[0],3), "$",
                                                                        "--REC:",round(rec_val[0], 3),'$\\pm',round(rec_error[0],3), "$",
                                                                       "-- acc:",round(acc_val[0], 3),'$\\pm',round(acc_error[0],3), "$",
                                                                        "--gmn:",round(gmn_val[0], 3),'$\\pm',round(gmn_error[0],3), "$","}", file=f_out)
    print("\label{tab:student}", file=f_out)



def latex_table_mcnemar(names, p_values,stats,rejects, area2s, prec2s, f1_2s, recall2s, acc2s, gmean2s,
                                                area1,  prec1,  f1_1, recall1,  acc1,  gmean1, f_out):

    print("\\begin{tabular}{c | c  c  c  c  c  c  c  c  c}\hline \hline", file=f_out)
    print("Model & $stats$  & p-val  &  Reject $H_{0}$ &  AUC  & PREC  &  F1S  & RECA  &  ACC  &  G-MEAN  \\\  \hline", file=f_out)

    print('model:-1',' & ', "--",' & ', "--",' & ', "--",  ' & ',
              round(area1,2),   ' & ', round(prec1,2), ' & ', round(f1_1,2),' & ',
              round(recall1,2),' & ', round(acc1,2),  ' & ', round(gmean1,2),
              ' \\\ ', file=f_out)

    nAlgos = len(p_values)
    reject_h0 = ''

    for i in range(nAlgos):                    
        if rejects[i]: reject_h0 = '\\checkmark'
        elif not rejects[i]: reject_h0 = '\\xmark'                
        print(names[i], ' & ',
              round(stats[i],3),   ' & ', round(p_values[i],3),' & ', reject_h0,        ' & ',
              round(area2s[i],2),  ' & ', round(prec2s[i],2), ' & ', round(f1_2s[i],2),' & ',
              round(recall2s[i],2),' & ', round(acc2s[i],2),  ' & ', round(gmean2s[i],2),
              ' \\\ ', file=f_out)

        
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\caption{Mc-Nemar statistics test. Scores of current classifier}", file=f_out)
    print("\label{tab:mcnemar}", file=f_out)
