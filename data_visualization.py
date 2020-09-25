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

def plot_roc_curve(TPR,FPR,sample,real,glob_local):

    if(real=='sorted'):
        TPR = np.sort(TPR,axis=None)
        FPR = np.sort(FPR,axis=None)

    area = auc(FPR,TPR)
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)'  % area, linestyle="-", marker="o")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve -' + sample)
    plt.legend(loc="lower right")
    plt.savefig('./plots/roc_curve_'+sample+'_'+real+'_'+glob_local+'.png')
    output = pd.DataFrame({'False positive rate': FPR,'True positive rate': TPR, 'Area': area})
    output.to_csv('output/' + sample +  '/' + 'BoostSVM_ROC.csv', index=False)
    plt.close()