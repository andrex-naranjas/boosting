#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# data preparation module
# python basics
import sys

# data analysis and wrangling
import pandas as pd

# uproot to import ROOT format data
import uproot
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

# machine learning
from sklearn.model_selection import train_test_split

# bootstrap
from sklearn.utils import resample

# data visualization module
import data_visualization as dv


class data_preparation:

    def __init__(self):
        dummy = 0

    # fetch data
    def fetch_data(self, sample):
        if sample == 'titanic':
            data_set = pd.read_csv('./data/titanic.csv')
        elif sample == 'cancer':
            data_set = pd.read_csv('./data/breast_cancer.csv')
        elif sample == 'german':
            data_set = pd.read_csv('./data/german.csv')
        elif sample == 'heart':
            data_set = pd.read_csv('./data/heart.csv')
        elif sample == 'solar':
            data_set = pd.read_csv('./data/solar.csv')
        elif sample == 'car':
            data_set = pd.read_csv('./data/car.csv')
        elif sample == 'contra':
            data_set = pd.read_csv('./data/contra.csv')
        elif sample == 'tac_toe':
            data_set = pd.read_csv('./data/tac_toe.csv')
        elif sample == 'belle2_i':
            file = uproot.open('./data/belle2_kpipi0.root')
            data_set = file['combined'].pandas.df()
        elif sample == 'belle2_ii':
            file = uproot.open('./data/belle2_kpi.root')
            data_set = file['combined'].pandas.df()
        elif sample == 'belle2_iii':
            file_train = uproot.open('./data/train_D02k3pi.root')
            data_train = file_train['d0tree'].pandas.df()
            file_test  = uproot.open('./data/test_D02k3pi.root')
            data_test  = file_train['d0tree'].pandas.df()
            return data_train, data_test
        else:
            sys.exit('The sample name provided does not exist. Try again!')
        return data_set

    # call data
    def dataset(self, sample_name='', data_set=None, data_train=None, data_test=None,
                sampling=False, split_sample=0, train_test=False):
        
        # if sampling = True, sampling is done outside data_preparation,
        # sample is fetched externally

        # fetch data_set if NOT externally provided
        if not sampling:
            if not train_test:
                data_set = self.fetch_data(sample_name)
            else: # there is separte data samples for training and testing
                data_train,data_test = self.fetch_data(sample_name)
            
        # prepare data
        if sample_name == 'titanic':
            X,Y = self.titanic(data_set)
        elif sample_name == 'cancer':
            X,Y = self.bCancer(data_set)
        elif sample_name == 'german':
            X,Y = self.german(data_set)
        elif sample_name == 'heart':
            X,Y = self.heart(data_set)
        elif sample_name == 'solar':
            X,Y = self.solar(data_set)
        elif sample_name == 'car':
            X,Y = self.car(data_set)
        elif sample_name == 'contra':
            X,Y = self.contra(data_set)
        elif sample_name == 'tac_toe':
            X,Y = self.tac_toe(data_set)
        elif sample_name == 'belle2_i' or sample_name == 'belle2_ii':
            X,Y = self.belle2(data_set, sampling, sample_name=sample_name)
        elif sample_name == 'belle2_iii':
            X_train, Y_train, X_test, Y_test = self.belle2_3pi(data_train, data_test, sampling, sample_name=sample_name)

        # print data after preparation
        print("After preparation shapes X and Y")#, X.shape, Y.shape)
        if(sample_name!='belle2_iii'): print(X.head())#, Y.head())
        if(sample_name!='belle2_iii'): print(Y.head())#, Y.head())
        if(sample_name=='belle2_iii'): print(X_train.head())#, Y.head())
        if(sample_name=='belle2_iii'): print(Y_train.head())#, Y.head())
                                  
        # divide sample into train and test sample
        if not train_test:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_sample)
        
        return X_train, Y_train, X_test, Y_test

    # belle2 data preparation
    def belle2(self, data_set, sampling, sample_name):

        # column names list
        cols = list(data_set.columns)

        if(sampling): # sampling was already carried, don't sample again!
            Y = data_set["Class"]
            # Data scaling [0,1]
            data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),columns = cols)
            X = data_set.drop("Class", axis=1)            
            return X,Y 

        sampled_data = resample(data_set, replace = False, n_samples = 3500, random_state = 0)

        Y = sampled_data["Class"]
        # Data scaling [0,1]
        # sampled_data = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data),columns = cols)

        # plot variables for visualization (now only for HEP)
        dv.plot_hist_frame(data_set,'full_'+sample_name)
        dv.plot_hist_frame(sampled_data,'sampled_'+sample_name)

        X = sampled_data.drop("Class", axis=1)
        return X,Y


    # belle2 data preparation
    def belle2_3pi(self, data_train, data_test, sampling, sample_name):

        # change value labels
        title_mapping = {0: -1, 1: 1}
        data_train['isSignal'] = data_train['isSignal'].map(title_mapping)
        data_train['isSignal'] = data_train['isSignal'].fillna(0)        
        data_test['isSignal']  = data_test['isSignal'].map(title_mapping)
        data_test['isSignal']  = data_test['isSignal'].fillna(0)

        if(sampling): # sampling was already carried, don't sample again!
            Y_train = data_train["isSignal"]
            Y_test  = data_test["isSignal"]
            # Data scaling [0,1]
            cols = list(data_train.columns)        
            data_train = pd.DataFrame(MinMaxScaler().fit_transform(data_train),columns = cols)
            data_train = data_train.drop("M", axis=1)            

            data_test  = pd.DataFrame(MinMaxScaler().fit_transform(data_test) ,columns = cols)
            data_test  = data_test.drop("M", axis=1)
            
            X_test  = data_test.drop("isSignal", axis=1)
            X_train = data_train.drop("isSignal", axis=1)
            return X_train, Y_train, X_test, Y_test

        sampled_data_train = resample(data_train, replace = True, n_samples = 1000, random_state=0)
        sampled_data_test  = resample(data_test,  replace = False, n_samples = 10000, random_state=0)

        Y_train = sampled_data_train["isSignal"]
        Y_test  = sampled_data_test["isSignal"]

        sampled_data_train = sampled_data_train.drop("M", axis=1)
        sampled_data_test  = sampled_data_test.drop("M", axis=1)

        # column names list
        cols = list(sampled_data_train.columns)
        
        # data scaling [0,1]
        sampled_data_train = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_train),columns = cols)
        sampled_data_test  = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_test), columns = cols)
        
        X_train = sampled_data_train.drop("isSignal", axis=1)
        X_test  = sampled_data_test.drop("isSignal", axis=1)
        return X_train, Y_train, X_test, Y_test


    #Titanic data preparation
    def titanic(self, data_set):

        # column names list
        cols = list(data_set.columns)

        data_set['Title'] = data_set.Name.str.extract('([A-Za-z]+)', expand=False)
        data_set = data_set.drop(['Name'], axis=1)

        #change names
        data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data_set['Title'] = data_set['Title'].replace('Mlle', 'Miss')
        data_set['Title'] = data_set['Title'].replace('Ms', 'Miss')
        data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data_set['Title'] = data_set['Title'].map(title_mapping)
        data_set['Title'] = data_set['Title'].fillna(0)

        #transform sex
        data_set['Sex'] = data_set['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        #group/transforming ages
        data_set.loc[ data_set['Age'] <= 16, 'Age'] = 0
        data_set.loc[(data_set['Age'] > 16) & (data_set['Age'] <= 32), 'Age'] = 1
        data_set.loc[(data_set['Age'] > 32) & (data_set['Age'] <= 48), 'Age'] = 2
        data_set.loc[(data_set['Age'] > 48) & (data_set['Age'] <= 64), 'Age'] = 3
        data_set.loc[ data_set['Age'] > 64, 'Age'] = 4

        #combine and drop features
        data_set['FamilySize'] = data_set['Siblings/Spouses Aboard'] + data_set['Parents/Children Aboard'] + 1
        data_set = data_set.drop(['Siblings/Spouses Aboard'], axis=1)
        data_set = data_set.drop(['Parents/Children Aboard'], axis=1)

        #create a new feature(s)
        data_set['IsAlone'] = 0
        data_set.loc[data_set['FamilySize'] == 1, 'IsAlone'] = 1

        #return data_set #(for tmva prep)

        #change names
        title_mapping = {0: 1, 1: -1}
        data_set['Survived'] = data_set['Survived'].map(title_mapping)
        data_set['Survived'] = data_set['Survived'].fillna(0)

        Y = data_set["Survived"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), columns = cols)

        X = data_set.drop("Survived", axis=1)
        return X,Y

    #Breast-cancer data preparation
    def bCancer(self, data_set):

        # column names list
        cols = list(data_set.columns)

        #change names
        title_mapping = {"no-recurrence-events": 1, "recurrence-events": -1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        title_mapping = {'10-19': 0, '20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60-69': 5, '70-79': 6, '80-89': 7, '90-99': 8}
        data_set['age'] = data_set['age'].map(title_mapping)
        data_set['age'] = data_set['age'].fillna(0)

        title_mapping = {'lt40': 0, 'ge40': 1, 'premeno': 2}
        data_set['menopause'] = data_set['menopause'].map(title_mapping)
        data_set['menopause'] = data_set['menopause'].fillna(0)

        title_mapping = {'0-4': 0,'5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10, '55-59': 11}
        data_set['tumorSize'] = data_set['tumorSize'].map(title_mapping)
        data_set['tumorSize'] = data_set['tumorSize'].fillna(0)

        title_mapping = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '18-20': 6, '21-23': 7, '24-26': 8, '27-29': 9, '30-32': 10, '33-35': 11, '36-39': 12}
        data_set['invNodes'] = data_set['invNodes'].map(title_mapping)
        data_set['invNodes'] = data_set['invNodes'].fillna(0)

        title_mapping = {'yes': 0, 'no': 1}
        data_set['nodeCaps'] = data_set['nodeCaps'].map(title_mapping)
        data_set['nodeCaps'] = data_set['nodeCaps'].fillna(0)

        title_mapping ={'left': 0, 'right': 1}
        data_set['breast'] = data_set['breast'].map(title_mapping)
        data_set['breast'] = data_set['breast'].fillna(0)

        title_mapping = {'left-up': 0, 'left-low': 1, 'right-up': 2, 'right-low': 3, 'central': 4}
        data_set['breastQuad'] = data_set['breastQuad'].map(title_mapping)
        data_set['breastQuad'] = data_set['breastQuad'].fillna(0)

        title_mapping = {'yes': 0, 'no': 1}
        data_set['irradiat'] = data_set['irradiat'].map(title_mapping)
        data_set['irradiat'] = data_set['irradiat'].fillna(0)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), columns = cols)

        X = data_set.drop("Class", axis=1)
        return X,Y

    def two_norm(self, data_set):
        #These data are already well-formatted

        #return data_set #(for tmva data prep)
        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    def german(self, data_set):

        # column names list
        cols = list(data_set.columns)

        #change names
        title_mapping = {1: 1, 2: -1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        title_mapping = {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}
        data_set['Status'] = data_set['Status'].map(title_mapping)
        data_set['Status'] = data_set['Status'].fillna(0)

        title_mapping = {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}
        data_set['History'] = data_set['History'].map(title_mapping)
        data_set['History'] = data_set['History'].fillna(0)

        title_mapping = {'A40': 0, 'A41': 1, 'A42': 2, 'A43': 3, 'A44': 4,'A45': 5, 'A46': 6, 'A47': 7, 'A48': 8, 'A49': 9, 'A410': 10}
        data_set['Purpose'] = data_set['Purpose'].map(title_mapping)
        data_set['Purpose'] = data_set['Purpose'].fillna(0)

        title_mapping = {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}
        data_set['Savings'] = data_set['Savings'].map(title_mapping)
        data_set['Savings'] = data_set['Savings'].fillna(0)

        title_mapping = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
        data_set['Employment'] = data_set['Employment'].map(title_mapping)
        data_set['Employment'] = data_set['Employment'].fillna(0)

        title_mapping = {'A91': 0, 'A92': 1, 'A93': 2, 'A94': 3, 'A95': 4}
        data_set['SexStatus'] = data_set['SexStatus'].map(title_mapping)
        data_set['SexStatus'] = data_set['SexStatus'].fillna(0)

        title_mapping = {'A101': 0, 'A102': 1, 'A103': 2}
        data_set['Debtor'] = data_set['Debtor'].map(title_mapping)
        data_set['Debtor'] = data_set['Debtor'].fillna(0)

        title_mapping = {'A121': 0, 'A122': 1, 'A123': 2, 'A124': 3}
        data_set['Property'] = data_set['Property'].map(title_mapping)
        data_set['Property'] = data_set['Property'].fillna(0)

        title_mapping = {'A141': 0, 'A142': 1, 'A143': 2, 'A144': 3}
        data_set['Plans'] = data_set['Plans'].map(title_mapping)
        data_set['Plans'] = data_set['Plans'].fillna(0)

        title_mapping = {'A151': 0, 'A152': 1, 'A153': 2}
        data_set['Housing'] = data_set['Housing'].map(title_mapping)
        data_set['Housing'] = data_set['Housing'].fillna(0)

        title_mapping = {'A171': 0, 'A172': 1, 'A173': 2,  'A174': 3}
        data_set['Job'] = data_set['Job'].map(title_mapping)
        data_set['Job'] = data_set['Job'].fillna(0)

        title_mapping = {'A191': 0, 'A192': 1}
        data_set['Phone'] = data_set['Phone'].map(title_mapping)
        data_set['Phone'] = data_set['Phone'].fillna(0)

        title_mapping = {'A201': 0, 'A202': 1}
        data_set['Foreign'] = data_set['Foreign'].map(title_mapping)
        data_set['Foreign'] = data_set['Foreign'].fillna(0)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(
                                MinMaxScaler().fit_transform(data_set),
                                columns = cols
                                )

        X = data_set.drop("Class", axis=1)

        return X,Y

    # heart conditions data set
    def heart(self, data_set):

        # column names list
        cols = list(data_set.columns)

        #change names
        title_mapping = {0: 1, 1: -1, 2: -1, 3: -1, 4: -1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(
                                MinMaxScaler().fit_transform(data_set),
                                columns = cols
                                )

        X = data_set.drop("Class", axis=1)
        return X,Y


    def solar(self, data_set):

        #change names
        title_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'H': 6}
        data_set['Zurich'] = data_set['Zurich'].map(title_mapping)
        data_set['Zurich'] = data_set['Zurich'].fillna(0)

        title_mapping = {'X': 0, 'R': 1, 'S': 2, 'A': 3, 'H': 4, 'K': 5}
        data_set['Size'] = data_set['Size'].map(title_mapping)
        data_set['Size'] = data_set['Size'].fillna(0)

        title_mapping = {'X': 0, 'O': 1, 'I': 2, 'C': 3}
        data_set['Distro'] = data_set['Distro'].map(title_mapping)
        data_set['Distro'] = data_set['Distro'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Activity'] = data_set['Activity'].map(title_mapping)
        data_set['Activity'] = data_set['Activity'].fillna(0)

        title_mapping = {1: 0, 2: 1, 3: 2}
        data_set['Evolution'] = data_set['Evolution'].map(title_mapping)
        data_set['Evolution'] = data_set['Evolution'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Prev24'] = data_set['Prev24'].map(title_mapping)
        data_set['Prev24'] = data_set['Prev24'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Histo'] = data_set['Histo'].map(title_mapping)
        data_set['Histo'] = data_set['Histo'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Complex'] = data_set['Complex'].map(title_mapping)
        data_set['Complex'] = data_set['Complex'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Area'] = data_set['Area'].map(title_mapping)
        data_set['Area'] = data_set['Area'].fillna(0)

        title_mapping = {1: 0, 2: 1}
        data_set['Largest'] = data_set['Largest'].map(title_mapping)
        data_set['Largest'] = data_set['Largest'].fillna(0)

        #create the Class
        data_set['Class']=data_set.sum(axis=1)
        data_set.loc[(data_set['Class1'] == 0) & (data_set['Class2'] == 0) & (data_set['Class3'] == 0), 'Class'] = 1
        data_set.loc[(data_set['Class1'] != 0) | (data_set['Class2'] != 0) | (data_set['Class3'] != 0), 'Class'] = -1
        data_set = data_set.drop(['Class1', 'Class2', 'Class3' ], axis=1)

        # column names list
        cols = list(data_set.columns)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(
                                MinMaxScaler().fit_transform(data_set),
                                columns = cols
                                )

        X = data_set.drop("Class", axis=1)
        return X,Y


    def car(self, data_set):

        # column names list
        cols = list(data_set.columns)

        title_mapping = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
        data_set['Buy'] = data_set['Buy'].map(title_mapping)
        data_set['Buy'] = data_set['Buy'].fillna(0)

        title_mapping = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
        data_set['Maint'] = data_set['Maint'].map(title_mapping)
        data_set['Maint'] = data_set['Maint'].fillna(0)

        title_mapping = {2: 0, 3: 1, 4: 2, '5more': 3}
        data_set['Doors'] = data_set['Doors'].map(title_mapping)
        data_set['Doors'] = data_set['Doors'].fillna(0)

        title_mapping = {2: 0, 4: 1, 'more': 2}
        data_set['Persons'] = data_set['Persons'].map(title_mapping)
        data_set['Persons'] = data_set['Persons'].fillna(0)

        title_mapping = {2: 0, 4: 1, 'more': 2}
        data_set['Persons'] = data_set['Persons'].map(title_mapping)
        data_set['Persons'] = data_set['Persons'].fillna(0)

        title_mapping = {'small': 0, 'med': 1, 'big': 2}
        data_set['Lug'] = data_set['Lug'].map(title_mapping)
        data_set['Lug'] = data_set['Lug'].fillna(0)

        title_mapping = {'low': 0, 'med': 1, 'high': 2}
        data_set['Safety'] = data_set['Safety'].map(title_mapping)
        data_set['Safety'] = data_set['Safety'].fillna(0)

        title_mapping = {'unacc': -1, 'acc': 1, 'good': 1, 'vgood': 1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(
                                MinMaxScaler().fit_transform(data_set),
                                columns = cols
                                )

        X = data_set.drop("Class", axis=1)
        return X,Y


    def contra(self, data_set):

        data_set.loc[ data_set['Age'] <= 17, 'Age'] = 0
        data_set.loc[(data_set['Age'] > 17) & (data_set['Age'] <= 25), 'Age'] = 1
        data_set.loc[(data_set['Age'] > 25) & (data_set['Age'] <= 33), 'Age'] = 2
        data_set.loc[(data_set['Age'] > 33) & (data_set['Age'] <= 41), 'Age'] = 3
        data_set.loc[ data_set['Age'] > 41, 'Age'] = 4

        title_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        data_set['W_edu'] = data_set['W_edu'].map(title_mapping)
        data_set['W_edu'] = data_set['W_edu'].fillna(0)

        title_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        data_set['H_edu'] = data_set['H_edu'].map(title_mapping)
        data_set['H_edu'] = data_set['H_edu'].fillna(0)

        ###Children missing, se puede categorizar en intervalos?

        ##W_religion OKAY
        ##W_work OKAY

        title_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        data_set['H_occupation'] = data_set['H_occupation'].map(title_mapping)
        data_set['H_occupation'] = data_set['H_occupation'].fillna(0)

        title_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        data_set['Living_index'] = data_set['Living_index'].map(title_mapping)
        data_set['Living_index'] = data_set['Living_index'].fillna(0)

        ##Exposure OKAY

        title_mapping = {1: -1, 2: 1, 3: 1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        data_set = data_set.drop('Children',axis=1)

        # column names list
        cols = list(data_set.columns)

        Y = data_set["Class"]

        # Data scaling [0,1]
        data_set = pd.DataFrame(
                                MinMaxScaler().fit_transform(data_set),
                                columns = cols
                                )

        X = data_set.drop("Class", axis=1)
        return X,Y


    def nursery(self, data_set):

        title_mapping = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
        data_set['Parents'] = data_set['Parents'].map(title_mapping)
        data_set['Parents'] = data_set['Parents'].fillna(0)

        title_mapping = {'proper' : 0, 'less_proper' : 1, 'improper' : 2, 'critical' : 3, 'very_crit' : 4}
        data_set['Has_nurs'] = data_set['Has_nurs'].map(title_mapping)
        data_set['Has_nurs'] = data_set['Has_nurs'].fillna(0)

        title_mapping = {'complete' : 0, 'completed' : 1, 'incomplete' : 2, 'foster' : 3}
        data_set['Form_fam'] = data_set['Form_fam'].map(title_mapping)
        data_set['Form_fam'] = data_set['Form_fam'].fillna(0)

        title_mapping = {1: 0, 2: 1, 3: 2, 'more': 4}
        data_set['Children'] = data_set['Children'].map(title_mapping)
        data_set['Children'] = data_set['Children'].fillna(0)

        title_mapping = {'convenient' : 0, 'less_conv' : 1, 'critical' : 2}
        data_set['Housing'] = data_set['Housing'].map(title_mapping)
        data_set['Housing'] = data_set['Housing'].fillna(0)

        title_mapping = {'convenient' : 0, 'inconv' : 0}
        data_set['Finance'] = data_set['Finance'].map(title_mapping)
        data_set['Finance'] = data_set['Finance'].fillna(0)

        title_mapping = {'nonprob' : 0, 'slightly_prob' : 1, 'problematic' : 2}
        data_set['Social'] = data_set['Social'].map(title_mapping)
        data_set['Social'] = data_set['Social'].fillna(0)

        title_mapping = {'recommended' : 0, 'priority' : 1, 'not_recom' : 2}
        data_set['Health'] = data_set['Health'].map(title_mapping)
        data_set['Health'] = data_set['Health'].fillna(0)

        title_mapping = {'not_recom' : -1, 'recommend' : 1, 'very_recom' : 1, 'priority' : 1, 'spec_prior' : 1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)

        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    def tac_toe(self, data_set):

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['TL'] = data_set['TL'].map(title_mapping)
        data_set['TL'] = data_set['TL'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['TM'] = data_set['TM'].map(title_mapping)
        data_set['TM'] = data_set['TM'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['TR'] = data_set['TR'].map(title_mapping)
        data_set['TR'] = data_set['TR'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['ML'] = data_set['ML'].map(title_mapping)
        data_set['ML'] = data_set['ML'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['MM'] = data_set['MM'].map(title_mapping)
        data_set['MM'] = data_set['MM'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['MR'] = data_set['MR'].map(title_mapping)
        data_set['MR'] = data_set['MR'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['BL'] = data_set['BL'].map(title_mapping)
        data_set['BL'] = data_set['BL'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['BM'] = data_set['BM'].map(title_mapping)
        data_set['BM'] = data_set['BM'].fillna(0)

        title_mapping = {'x': 0, 'o': 1, 'b': 2}
        data_set['BR'] = data_set['BR'].map(title_mapping)
        data_set['BR'] = data_set['BR'].fillna(0)

        title_mapping = {'positive' : 1, 'negative' : -1}
        data_set['Class'] = data_set['Class'].map(title_mapping)
        data_set['Class'] = data_set['Class'].fillna(0)


        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    # bin data?!?
    # Xin = data_set.drop("Class", axis=1)
    # est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    # est.fit(Xin)
    
    # XT = est.transform(Xin)
    
    # # Creating pandas dataframe from numpy array
    # X = pd.DataFrame({'D0_m': XT[:, 0], 'D0_p': XT[:, 1], 'p0_p': XT[:, 2]})
