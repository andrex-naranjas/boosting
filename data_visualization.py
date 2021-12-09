'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# visualization module
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import pandas as pd
import math as math
from sklearn.metrics import auc
import csv

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

    plot_tile=''
    if my_kernel == 'rbf':
        sigmax,sigmin=100,0.00
        cmax,cmin=100,0
        plot_tile='RBF'        
    elif my_kernel == 'sigmoid':
        sigmax,sigmin=0.1,0.00
        cmax,cmin=100,0
        plot_tile='Sigmoid'
    elif my_kernel == 'poly' or my_kernel == 'linear':
        sigmax,sigmin=0.1,0.00
        cmax,cmin=10,0
        if my_kernel == 'poly':   plot_tile='Polynomial'
        if my_kernel == 'linear': plot_tile='Linear'


    # tick_x = [math.floor(sigmax), math.floor(sigmax/2) , math.floor(sigmin)]
    # tick_y = [math.floor(cmax),math.floor(cmax/2),math.floor(cmin)]
    half_sig = sigmax/2
    if my_kernel =='rbf': half_sig = int(half_sig)
    
    tick_x = [sigmax, half_sig, int(sigmin)]
    tick_y = [cmax,   int(cmax/2),   cmin]

    matrix = 100*matrix
    matrix = matrix.astype(int)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    
    ax.set_yticklabels(tick_y)
    ax.yaxis.label.set_size(15)
    
    ax.set_xticklabels(tick_x)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.xaxis.label.set_size(15)
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$C$')
    
    # ax.spines['left'].set_visible(False)
    # ax.set_xticks(np.arange(matrix.shape[1])) # show all ticks
    # ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks([0,matrix.shape[1]/2, matrix.shape[1]-1])
    ax.set_yticks([0,matrix.shape[0]/2, matrix.shape[0]-1])

    # loop over data dimensions and create text annotations.
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            text = ax.text(j, i, math.floor(matrix[i,j]),
                           ha="center", va="center", color="black")

    #ax.set_title('Test Error (%) - '+sample_name+' dataset')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator
    #ax_int = ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.locator_params(nbins=6)
    
    cbar.set_label('Train error %')
    plt.title(plot_tile, y=-0.1)

    plt.savefig('./plots/2dplot_'+sample_name+'.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close()


def plot_ordered_stats_summary(val_acc, val_auc, val_prc, names, sample_name, metric='auc'):

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.275, top=0.99, right=0.99, left=0.085)
    ax = fig.add_subplot(111)
    size_x = len(val_auc)
    
    x = np.linspace(0, size_x, size_x)
    y1 = val_acc
    y2 = val_auc
    y3 = val_prc
    ax.set_xticks(x)
    #ax.set_ylim([-0.05,1.05])
    ax.set_xticklabels(names, rotation=90, ha='center', fontsize=7)

    ax.scatter(x,y1, c='b', marker="^", label='ACC')
    ax.scatter(x,y2, c='r', marker="+", label='AUC')
    ax.scatter(x,y3, c='k', marker=".", label='PRC')
    at = AnchoredText(sample_name, prop=dict(size=12.5), frameon=False, loc='lower left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.axhline(y=np.mean(y2), color="r", linewidth=0.75, linestyle="--")
    plt.axhline(y=np.mean(y3), color="k", linewidth=0.75, linestyle="--")
    #plt.plot(x, 0.95, '-k')

    plt.legend(loc=4, fancybox=False, title_fontsize='medium')
    
    #plt.rcParams["figure.figsize"] = (20,30)    
    # plt.text(0.15, 0.9,'$\\mu$={}, $\\sigma$={}'.format(round(np.mean(sample),1), round(np.std(sample),4)),
    #          ha='center', va='center', transform=ax.transAxes)
    # plt.text(0.15, 0.8,'$p_{{val}}$={}, $\\alpha$={}'.format(round(p,3), alpha),
    #          ha='center', va='center', transform=ax.transAxes)
    
    #plt.xlabel(xlabel)
    plt.ylabel('Metric A.U.')

    plt.savefig('./plots/rank_'+metric+'_'+sample_name+'.pdf')
    plt.close()


def save_df_selected_classifiers(mean_list_auc, mean_list_acc, mean_list_prc, name_list, f_names, sample_name):
    # save_df_selected_classifiers(mean_list_acc, mean_list_auc, mean_list_prc, name_list, flavor_names, sample_name) actual calling
    # select the best 10 AUC, with the requirement that the ACC and PRC are above average
    thres_acc = np.mean(mean_list_acc)*0
    thres_prc = np.mean(mean_list_prc)*0
    selected_classifiers = []
    for i in range(len(mean_list_auc)):
        
        if mean_list_acc[i] > thres_acc and mean_list_prc[i] > thres_prc or True:
            selected_classifiers.append(name_list[i])

        if len(selected_classifiers) == 4:
            break
        
    total_selected_classifier_val = []
    for i in range(len(f_names)):

        fill_flag = True
        for j in range(len(selected_classifiers)):
            if f_names[i] == selected_classifiers[j]:
                total_selected_classifier_val.append(1)
                fill_flag = False
                break

        if fill_flag:
            total_selected_classifier_val.append(0)

    
    col_val = pd.DataFrame(data=np.array(total_selected_classifier_val), columns=[sample_name])
    col_nam = pd.DataFrame(data=np.array(f_names), columns=["classifier"])
    df = pd.concat([col_nam["classifier"], col_val[sample_name]],
                   axis=1, keys=["classifier", sample_name])

    df.to_csv(str('./tables/CSV/'+sample_name+'_selected_classifier.csv'), index=False)
    

def voting_table():
    # create voting table to decide which classifiers go to the olympics
    titanic =  pd.read_csv("./tables/CSV/titanic_selected_classifier.csv")["titanic"]
    cancer  =  pd.read_csv("./tables/CSV/cancer_selected_classifier.csv")["cancer"]
    german  =  pd.read_csv("./tables/CSV/german_selected_classifier.csv")["german"]
    heart   =  pd.read_csv("./tables/CSV/heart_selected_classifier.csv")["heart"]
    solar   =  pd.read_csv("./tables/CSV/solar_selected_classifier.csv")["solar"]
    car     =  pd.read_csv("./tables/CSV/car_selected_classifier.csv")["car"]
    ecoli   =  pd.read_csv("./tables/CSV/ecoli_selected_classifier.csv")["ecoli"]
    wine    =  pd.read_csv("./tables/CSV/wine_selected_classifier.csv")["wine"]
    abalone =  pd.read_csv("./tables/CSV/abalone_selected_classifier.csv")["abalone"]
    names   =  pd.read_csv("./tables/CSV/abalone_selected_classifier.csv")["classifier"]

    total = titanic + cancer + german + heart + solar + car + ecoli + wine + abalone

    df = pd.concat([names, titanic, cancer, german, heart, solar,  car, ecoli, wine, abalone, total], axis=1,
                   keys=["classifier","titanic","cancer","german","heart","solar","car","ecoli","wine","abalone","total"])
    
    df.to_csv("./tables/CSV/rank_voting.csv")

    # latex table
    f_out = open('./tables/rank_voting.tex', "w")
    
    print("\\begin{tabular}{c | c  c  c  c c  c  c  c c  c  c  c}\hline \hline", file=f_out)
    print("Model &  Titanic  &  Cancer  &  German  &  Heart  & Solar & Car & Ecoli  &  Wine  & Abalone & Adult  & Connect & Total  \\\  \hline", file=f_out)

    for k in range(len(titanic)): # loop number of ensembles
        titanic_v, cancer_v, german_v, heart_v, solar_v = '\\checkmark','\\checkmark','\\checkmark','\\checkmark','\\checkmark'
        car_v, ecoli_v, wine_v, abalone_v, adult_v, connect_v  = '\\checkmark','\\checkmark','\\checkmark','\\checkmark','\\checkmark','\\checkmark'

        if titanic[k] == 0: titanic_v  = '\\xmark'
        if cancer[k]  == 0: cancer_v   = '\\xmark'
        if german[k]  == 0: german_v   = '\\xmark'
        if heart[k]   == 0: heart_v    = '\\xmark'
        if solar[k]   == 0: solar_v    = '\\xmark'
        if car[k]     == 0: car_v      = '\\xmark'
        if ecoli[k]   == 0: ecoli_v    = '\\xmark'
        if wine[k]    == 0: wine_v     = '\\xmark'
        if abalone[k] == 0: abalone_v  = '\\xmark'
        if 0 == 0: adult_v    = '\\xmark'
        if 0 == 0: connect_v  = '\\xmark'
        
        print(names[k], '&', titanic_v, '&', cancer_v, '&', german_v, '&', heart_v, '&', solar_v, '&', car_v, '&',
              ecoli_v, '&', wine_v, '&', abalone_v, '&', adult_v, '&', connect_v, '&', total[k], ' \\\ ', file=f_out)
                        
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\label{tab:student}", file=f_out)

    f_out.close()

    
    
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
                                                                        "--acc:",round(acc_val[0], 3),'$\\pm',round(acc_error[0],3), "$",
                                                                        "--gmn:",round(gmn_val[0], 3),'$\\pm',round(gmn_error[0],3), "$","}", file=f_out)
    print("\label{tab:student}", file=f_out)



def latex_table_student_single(names, sample, metric_name=None, metric_val=None, metric_error=None,
                               pvalues=None, mean_values=None, error_values=None, f_out=None):

    mean_err_1 = '$'+str(round(mean_values[0],2))+'\\!\\!\\pm\\!\\!'+str(round(error_values[0],2))+'$'
    mean_err_2 = '$'+str(round(mean_values[1],2))+'\\!\\!\\pm\\!\\!'+str(round(error_values[1],2))+'$'
    mean_err_3 = '$'+str(round(mean_values[2],2))+'\\!\\!\\pm\\!\\!'+str(round(error_values[2],2))+'$'
    mean_err_4 = '$'+str(round(mean_values[3],2))+'\\!\\!\\pm\\!\\!'+str(round(error_values[3],2))+'$'

    pval_1 = pvalues[0]
    pval_2 = pvalues[1]
    pval_3 = pvalues[2]
    pval_4 = pvalues[3]

    reject_1 = ''
    reject_2 = ''
    reject_3 = ''
    reject_4 = ''
    alpha = 0.05

    blue_1 = 0
    blue_2 = 0
    blue_3 = 0
    blue_4 = 0

    red_1 = 0
    red_2 = 0
    red_3 = 0
    red_4 = 0
    
    print("\\begin{tabular}{c  c | c  c  c  c}\hline \hline", file=f_out)
    print(sample,     "&         &  Ensemble1        &  Ensemble2         &  Ensemble3         &  Ensemble4         \\\  \hline", file=f_out)
    print(metric_name,"&         &", mean_err_1, "&", mean_err_2,  "&", mean_err_3,  "&", mean_err_4, " \\\  \hline", file=f_out)
    print("Model & $\mu_{mtrc}$  &  R.$H_{0}$ &  R.$H_{0}$ & R.$H_{0}$  &  R.$H_{0}$   \\\  \hline", file=f_out)

    for k in range(len(metric_val)-1): # loop number of algorithms
        j = k + 1
        if pval_1[j] < alpha:
            if metric_val[j] < mean_values[0]:
                blue_1+=1
                reject_1 = '\\textcolor{blue}{\\checkmark}'
            else:
                red_1+=1
                reject_1 = '\\textcolor{red}{\\checkmark}'        
        else:   reject_1 = '\\xmark'

        if pval_2[j] < alpha:
            if metric_val[j] < mean_values[1]:
                blue_2+=1
                reject_2 = '\\textcolor{blue}{\\checkmark}'
            else:
                red_2+=1
                reject_2 = '\\textcolor{red}{\\checkmark}'
        else:   reject_2 = '\\xmark'

        if pval_3[j] < alpha:
            if metric_val[j] < mean_values[2]:
                blue_3+=1
                reject_3 = '\\textcolor{blue}{\\checkmark}'
            else:
                red_3+=1
                reject_3 = '\\textcolor{red}{\\checkmark}'        
        else:   reject_3 = '\\xmark'

        if pval_4[j] < alpha:
            if metric_val[j] < mean_values[3]:
                blue_4+=1
                reject_4 = '\\textcolor{blue}{\\checkmark}'
            else:
                red_4+=1
                reject_4 = '\\textcolor{red}{\\checkmark}'        
        else:   reject_4 = '\\xmark'
        
        
        print(names[j], '&',
              '$', str(round(metric_val[j],2))+'\\!\\!\\pm\\!\\!'+ str(round(metric_error[j],2)), "$",'&', reject_1, '&', reject_2,' & ',
              reject_3, '&', reject_4, ' \\\ ', file=f_out)
                        
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
    print("\label{tab:student}", file=f_out)

    # save total count to a CSV files
    header_csv = ['blue_1', 'red_1', 'blue_2', 'red_2', 'blue_3', 'red_3', 'blue_4', 'red_4', 'sample', 'metric_name']
    data_csv   = [blue_1, red_1, blue_2, red_2, blue_3, red_3, blue_4, red_4, sample, metric_name]
    with open('./tables/CSV/blue_red_'+sample+'_'+metric_name+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_csv)
        writer.writerow(data_csv)



def avarage_table_studente(metric='AUC'):

    titanic = pd.read_csv('./tables/CSV/blue_red_titanic_'+metric+'.csv')
    cancer  = pd.read_csv('./tables/CSV/blue_red_cancer_'+metric+'.csv')
    german  = pd.read_csv('./tables/CSV/blue_red_german_'+metric+'.csv')
    heart   = pd.read_csv('./tables/CSV/blue_red_heart_'+metric+'.csv')
    car     = pd.read_csv('./tables/CSV/blue_red_car_'+metric+'.csv')
    ecoli   = pd.read_csv('./tables/CSV/blue_red_ecoli_'+metric+'.csv')
    wine    = pd.read_csv('./tables/CSV/blue_red_wine_'+metric+'.csv')
    abalone = pd.read_csv('./tables/CSV/blue_red_abalone_'+metric+'.csv')

    #     blue_1,red_1,blue_2,red_2,blue_3,red_3,blue_4,red_4,sample,metric_name
    # 10,0,0,17,7,0,4,0,ecoli,ACC

    blue_ensamble_1 = titanic['blue_1'][0]+cancer['blue_1'][0]+german['blue_1'][0]+heart['blue_1'][0]+car['blue_1'][0]+ecoli['blue_1'][0]+wine['blue_1'][0]+abalone['blue_1'][0]
    blue_ensamble_2 = titanic['blue_2'][0]+cancer['blue_2'][0]+german['blue_2'][0]+heart['blue_2'][0]+car['blue_2'][0]+ecoli['blue_2'][0]+wine['blue_2'][0]+abalone['blue_2'][0]
    blue_ensamble_3 = titanic['blue_3'][0]+cancer['blue_3'][0]+german['blue_3'][0]+heart['blue_3'][0]+car['blue_3'][0]+ecoli['blue_3'][0]+wine['blue_3'][0]+abalone['blue_3'][0]
    blue_ensamble_4 = titanic['blue_4'][0]+cancer['blue_4'][0]+german['blue_4'][0]+heart['blue_4'][0]+car['blue_4'][0]+ecoli['blue_4'][0]+wine['blue_4'][0]+abalone['blue_4'][0]

    red_ensamble_1 = titanic['red_1'][0]+cancer['red_1'][0]+german['red_1'][0]+heart['red_1'][0]+car['red_1'][0]+ecoli['red_1'][0]+wine['red_1'][0]+abalone['red_1'][0]
    red_ensamble_2 = titanic['red_2'][0]+cancer['red_2'][0]+german['red_2'][0]+heart['red_2'][0]+car['red_2'][0]+ecoli['red_2'][0]+wine['red_2'][0]+abalone['red_2'][0]
    red_ensamble_3 = titanic['red_3'][0]+cancer['red_3'][0]+german['red_3'][0]+heart['red_3'][0]+car['red_3'][0]+ecoli['red_3'][0]+wine['red_3'][0]+abalone['red_3'][0]
    red_ensamble_4 = titanic['red_4'][0]+cancer['red_4'][0]+german['red_4'][0]+heart['red_4'][0]+car['red_4'][0]+ecoli['red_4'][0]+wine['red_4'][0]+abalone['red_4'][0]

    blue_1 = str(blue_ensamble_1)
    blue_2 = str(blue_ensamble_2)
    blue_3 = str(blue_ensamble_3) 
    blue_4 = str(blue_ensamble_4)
    
    red_1  = str(red_ensamble_1)
    red_2  = str(red_ensamble_2)  
    red_3  = str(red_ensamble_3)  
    red_4  = str(red_ensamble_4)

    titanic_string =''
    cancer_string  =''
    german_string  =''
    heart_string   =''
    car_string     =''
    ecoli_string   =''
    wine_string    =''
    abalone_string =''
    
    n_ensem=4
    for i in range(n_ensem):
        column = ''
        if i!=n_ensem-1: column ='&'
        titanic_string += ' '+str(titanic['blue_'+str(i+1)][0])+' & '+str(titanic['red_'+str(i+1)][0]) + column
        cancer_string  += ' '+str(cancer['blue_'+str(i+1)][0]) +' & '+str(cancer['red_'+str(i+1)][0])  + column
        german_string  += ' '+str(german['blue_'+str(i+1)][0]) +' & '+str(german['red_'+str(i+1)][0])  + column
        heart_string   += ' '+str(heart['blue_'+str(i+1)][0])  +' & '+str(heart['red_'+str(i+1)][0])   + column
        car_string     += ' '+str(car['blue_'+str(i+1)][0])    +' & '+str(car['red_'+str(i+1)][0])     + column
        ecoli_string   += ' '+str(ecoli['blue_'+str(i+1)][0])  +' & '+str(ecoli['red_'+str(i+1)][0])   + column
        wine_string    += ' '+str(wine['blue_'+str(i+1)][0]   )+' & '+str(wine['red_'+str(i+1)][0])    + column
        abalone_string += ' '+str(abalone['blue_'+str(i+1)][0])+' & '+str(abalone['red_'+str(i+1)][0]) + column

    f_out=open('./tables/blue_red_'+metric+'.tex', "w")
    print("\\begin{tabular}{c | c c | c c | c c | c c}\hline \hline", file=f_out)
    print( metric, "         &  Ensemble1    &    &  Ensemble2  &       &  Ensemble3   &      &  Ensemble4   &      \\\  \hline", file=f_out)
    print("Dataset   &   $N_{rej}^{+}$ & $N_{rej}^{-}$ & $N_{rej}^{+}$ & $N_{rej}^{-}$ & $N_{rej}^{+}$ & $N_{rej}^{-}$ & $N_{rej}^{+}$ & $N_{rej}^{-}$   \\\  \hline", file=f_out)
    
    print('titanic &', titanic_string + "\\\ ", file=f_out)
    print('cancer &',  cancer_string  + "\\\ ", file=f_out)
    print('german &',  german_string  + "\\\ ", file=f_out)
    print('heart &',   heart_string   + "\\\ ", file=f_out)
    print('car &',     car_string     + "\\\ ", file=f_out)
    print('ecoli &',   ecoli_string   + "\\\ ", file=f_out)
    print('wine &',    wine_string    + "\\\ ", file=f_out)
    print('abalone &', abalone_string + "\\\ ", file=f_out)

    print("\hline", file=f_out)
    print('Total &', blue_1+' & '+red_1+' & '+ blue_2+' & '+red_2+' & '+ blue_3+' & '+red_3+' & '+ blue_4+' &  '+red_4+' \\\ ', file=f_out)
                            
    print('\hline \hline', file=f_out)
    print('\end{tabular}', file=f_out)
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
