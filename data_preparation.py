#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

#Data preparation module

#Titanic data preparation
def titanic(data_set):
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

    X = data_set.drop("Survived", axis=1)
    Y = data_set["Survived"]

    return X,Y

#Breast-cancer data preparation
def bCancer(data_set):
    #change names    
    title_mapping = {"no-recurrence-events": 0, "recurrence-events": 1}
    data_set['Class'] = data_set['Class'].map(title_mapping)
    data_set['Class'] = data_set['Class'].fillna(0)
    
    title_mapping = {'10-19': 0, '20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60-69': 5, '70-79': 6, '80-89': 7, '90-99': 8}
    data_set['age'] = data_set['age'].map(title_mapping)
    data_set['age'] = data_set['age'].fillna(0)

    title_mapping = {'lt40': 0, 'ge40': 1, 'premeno': 2}
    data_set['menopause'] = data_set['menopause'].map(title_mapping)
    data_set['menopause'] = data_set['menopause'].fillna(0)
    
    title_mapping = {'0-4': 0,'5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10, '55-59': 11}
    data_set['tumor-size'] = data_set['tumor-size'].map(title_mapping)
    data_set['tumor-size'] = data_set['tumor-size'].fillna(0)
    
    title_mapping = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '18-20': 6, '21-23': 7, '24-26': 8, '27-29': 9, '30-32': 10, '33-35': 11, '36-39': 12}
    data_set['inv-nodes'] = data_set['inv-nodes'].map(title_mapping)
    data_set['inv-nodes'] = data_set['inv-nodes'].fillna(0)

    title_mapping = {'yes': 0, 'no': 1}
    data_set['node-caps'] = data_set['node-caps'].map(title_mapping)
    data_set['node-caps'] = data_set['node-caps'].fillna(0)
    
    title_mapping ={'left': 0, 'right': 1}
    data_set['breast'] = data_set['breast'].map(title_mapping)
    data_set['breast'] = data_set['breast'].fillna(0)
    
    title_mapping = {'left-up': 0, 'left-low': 1, 'right-up': 2, 'right-low': 3, 'central': 4}
    data_set['breast-quad'] = data_set['breast-quad'].map(title_mapping)
    data_set['breast-quad'] = data_set['breast-quad'].fillna(0)

    title_mapping = {'yes': 0, 'no': 1}
    data_set['irradiat'] = data_set['irradiat'].map(title_mapping)
    data_set['irradiat'] = data_set['irradiat'].fillna(0)

    X = data_set.drop("Class", axis=1)
    Y = data_set["Class"]
    
    return X,Y

def two_norm(data_set):
    #These data is already formatted
    X = data_set.drop("Class", axis=1)
    Y = data_set["Class"]
    return X,Y
