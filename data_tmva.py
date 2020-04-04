#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

# tmva module

from root_pandas import to_root

# data analysis and wrangling
import pandas as pd

# import module for data preparation
import data_preparation as dp

# fetch data set (from available list)
data_set = dp.fetch_data('heart')
new_data = dp.heart(data_set)

signal = new_data.loc[new_data['Class']==0]
background = new_data.loc[new_data['Class']==1]

signal.drop('Class', axis=1)
background.drop('Class', axis=1)

signal.to_root('signal.root', key='signal')
background.to_root('background.root', key='background')
