#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code to improve SVM
# authors: A. Ramirez-Morales and J. Salmon-Gamboa

# visualization module

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import math as math

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib
import math as math


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
    plt.savefig('./plots/'+name+'.pdf')
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

    ax.set_title('Test Error (%) '+sample_name)
    fig.tight_layout()
    plt.xlabel('ln $\sigma$')
    plt.ylabel('ln C')
    plt.savefig('2dplot_'+sample_name+'.pdf')
