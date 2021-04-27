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
from os import system,getenv,getuid,getcwd

# data analysis and wrangling
import pandas as pd
import numpy as np

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

workpath=getcwd()

class data_preparation:

    def __init__(self, GA_selection=False):
        self.genetic = GA_selection

    # fetch data
    def fetch_data(self, sample):
        if sample == "titanic":
            data_set = pd.read_csv(workpath+"/data/titanic.csv")
        elif sample == "cancer":
            data_set = pd.read_csv(workpath+"/data/breast_cancer.csv")
        elif sample == "german":
            data_set = pd.read_csv("./data/german.csv")
        elif sample == "heart":
            data_set = pd.read_csv("./data/heart.csv")
        elif sample == "solar":
            data_set = pd.read_csv("./data/solar.csv")
        elif sample == "car":
            data_set = pd.read_csv("./data/car.csv")
        elif sample == "ecoli":
            data_set = pd.read_csv("./data/ecoli.csv")
        elif sample == "wine":
            data_set = pd.read_csv("./data/wine.csv")
        elif sample == "abalone":
            data_set = pd.read_csv("./data/abalone.csv")
        elif sample == "adult":
            data_set = pd.read_csv("./data/adult.csv")
        elif sample == "connect":
            data_set = pd.read_csv("./data/connect.csv")            
        elif sample == "contra":
            data_set = pd.read_csv("./data/contra.csv")
        elif sample == "tac_toe":
            data_set = pd.read_csv("./data/tac_toe.csv")
        elif sample == "belle2_i":
            file = uproot.open("./data/belle2_kpipi0.root")            
            data_set = file["combined"].arrays(library="pd")
        elif sample == "belle2_ii":
            file = uproot.open("./data/belle2_kpi.root")
            data_set = file["combined"].arrays(library="pd")
        elif sample == "belle2_iii":
            file_train = uproot.open("./data/train_D02k3pi.root")
            data_train = file_train["d0tree"].arrays(library="pd")
            file_test  = uproot.open("./data/test_D02k3pi.root")
            data_test  = file_train["d0tree"].arrays(library="pd")
            return data_train, data_test
        else:
            sys.exit("The sample name provided does not exist. Try again!")
        return data_set

    # call data
    def dataset(self, sample_name, data_set=None, data_train=None, data_test=None,
                sampling=False, split_sample=0, train_test=False, indexes = None):
        
        # if sampling = True, sampling is done outside data_preparation,
        # sample is fetched externally

        # fetch data_set if NOT externally provided
        if not sampling:
            if not train_test:
                data_set = self.fetch_data(sample_name)
            else: # there is separate data samples for training and testing
                data_train,data_test = self.fetch_data(sample_name)
            
        # prepare data
        if sample_name == "titanic":
            X,Y = self.titanic(data_set)
        elif sample_name == "cancer":
            X,Y = self.bCancer(data_set)
        elif sample_name == "german":
            X,Y = self.german(data_set)
        elif sample_name == "heart":
            X,Y = self.heart(data_set)
        elif sample_name == "solar":
            X,Y = self.solar(data_set)
        elif sample_name == "car":
            X,Y = self.car(data_set)            
        elif sample_name == "ecoli":
            X,Y = self.ecoli(data_set)
        elif sample_name == "wine":
            X,Y = self.wine(data_set)
        elif sample_name == "abalone":
            X,Y = self.abalone(data_set)
        elif sample_name == "adult":
            X,Y = self.adult(data_set)
        elif sample_name == "connect":
            X,Y = self.connect(data_set)            
        elif sample_name == "contra":
            X,Y = self.contra(data_set)
        elif sample_name == "tac_toe":
            X,Y = self.tac_toe(data_set)
        elif sample_name == "belle2_i" or sample_name == "belle2_ii":
            X,Y = self.belle2(data_set, sampling, sample_name=sample_name)
        elif sample_name == "belle2_iii":
            X_train, Y_train, X_test, Y_test = self.belle2_3pi(data_train, data_test, sampling, sample_name=sample_name)

        # print data after preparation
        if not sampling:
            print("After preparation shapes X and Y")#, X.shape, Y.shape)
            if(sample_name!="belle2_iii"): print(X.head())#, Y.head())
            if(sample_name!="belle2_iii"): print(Y.head())#, Y.head())
            if(sample_name=="belle2_iii"): print(X_train.head())#, Y.head())
            if(sample_name=="belle2_iii"): print(Y_train.head())#, Y.head())

        # return X,Y without any spliting (for bootstrap)
        if sampling:
            return X,Y
                                  
        # divide sample into train and test sample
        if not train_test:
            if indexes is None:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_sample)
            else:
                X_train, X_test, Y_train, Y_test = self.indexes_split(X, Y, split_indexes=indexes)
                
        return X_train, Y_train, X_test, Y_test

    
    def indexes_split(self, X, Y, split_indexes):
        """ Function to split train and test data given train indexes"""
        total_indexes = np.array(X.index).tolist()        
        train_indexes = split_indexes.tolist()
        test_indexes  = list(set(total_indexes) - set(train_indexes))

        X_train = X.loc[train_indexes]
        Y_train = Y.loc[train_indexes]
        X_test  = X.loc[test_indexes]
        Y_test  = Y.loc[test_indexes]

        return X_train, X_test, Y_train, Y_test


    # belle2 data preparation
    def belle2(self, data_set, sampling, sample_name):
        
        if(sampling or self.genetic): # sampling was already carried, don"t sample again!
            Y = data_set["Class"]
            # Data scaling [0,1]
            # column names list            
            cols = list(data_set.columns)
            data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),columns = cols)
            X = data_set.drop("Class", axis=1)            
            return X,Y 

        sampled_data = resample(data_set, replace = False, n_samples = 5000, random_state = 0)

        Y = sampled_data["Class"]
        # column names list
        cols = list(sampled_data.columns)
        # Data scaling [0,1]
        #sampled_data = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data),columns = cols)
        X = sampled_data.drop("Class", axis=1)

        # # plot variables for visualization (now only for HEP)
        dv.plot_hist_frame(data_set,"full_"+sample_name)
        dv.plot_hist_frame(sampled_data,"sampled_"+sample_name)
            
        return X,Y


    # belle2 data preparation
    def belle2_3pi(self, data_train, data_test, sampling, sample_name):

        # change value labels
        title_mapping = {0: -1, 1: 1}
        data_train["isSignal"] = data_train["isSignal"].map(title_mapping)
        data_train["isSignal"] = data_train["isSignal"].fillna(0)        
        data_test["isSignal"]  = data_test["isSignal"].map(title_mapping)
        data_test["isSignal"]  = data_test["isSignal"].fillna(0)

        if(sampling or self.genetic): # sampling already done or not needed, don"t sample again!
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
        #return data_set #(for tmva prep)
        data_set = data_set.copy()        
        # set titles, these are quite problematic, so we format using the old way
        data_set.loc[:,"Title"] = data_set.Name.str.extract("([A-Za-z]+)", expand=False)
        data_set = data_set.drop(["Name"], axis=1)
        #change names
        data_set["Title"] = data_set["Title"].replace(["Lady", "Countess","Capt", "Col","Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
        data_set["Title"] = data_set["Title"].replace("Mlle", "Miss")
        data_set["Title"] = data_set["Title"].replace("Ms", "Miss")
        data_set["Title"] = data_set["Title"].replace("Mme", "Mrs")
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data_set["Title"] = data_set["Title"].map(title_mapping)
        data_set["Title"] = data_set["Title"].fillna(0)        

        #transform sex
        data_set.loc[data_set["Sex"]=="female", "Sex"]  = 0
        data_set.loc[data_set["Sex"]=="male",   "Sex"]  = 1

        #group/transforming ages
        data_set.loc[ data_set["Age"] <= 16, "Age"] = 0
        data_set.loc[(data_set["Age"] > 16) & (data_set["Age"] <= 32), "Age"] = 1
        data_set.loc[(data_set["Age"] > 32) & (data_set["Age"] <= 48), "Age"] = 2
        data_set.loc[(data_set["Age"] > 48) & (data_set["Age"] <= 64), "Age"] = 3
        data_set.loc[ data_set["Age"] > 64, "Age"] = 4

        #combine and drop features
        data_set["FamilySize"] = data_set["Siblings/Spouses Aboard"] + data_set["Parents/Children Aboard"] + 1
        data_set = data_set.drop(["Siblings/Spouses Aboard"], axis=1)
        data_set = data_set.drop(["Parents/Children Aboard"], axis=1)

        #create a new feature(s)
        data_set["IsAlone"] = 0
        data_set.loc[data_set["FamilySize"] == 1, "IsAlone"] = 1

        #change names
        title_mapping = {0: 1, 1: -1}
        #data_set["Survived"] = data_set["Survived"].map(title_mapping)
        #data_set["Survived"] = data_set["Survived"].fillna(0)
        data_set.loc[data_set["Survived"] == 0, "Survived"] = -1
        Y = data_set["Survived"]
        
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Survived", axis=1)
        return X,Y

    #Breast-cancer data preparation
    def bCancer(self, data_set):
        data_set = data_set.copy()
        # set the class names as +-1
        data_set.loc[data_set["Class"] == "no-recurrence-events", "Class"] = 1        
        data_set.loc[data_set["Class"] == "recurrence-events", "Class"] = -1
        
        # set age atributes as integers
        dummy = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
        for i in range(len(dummy)):
            data_set.loc[data_set["age"].astype(str) == dummy[i] , "age"] = i

        # set menopause atributtes as integers
        data_set.loc[data_set["menopause"]=="lt40",    "menopause"]  = 0
        data_set.loc[data_set["menopause"]=="ge40",    "menopause"]  = 1
        data_set.loc[data_set["menopause"]=="premeno", "menopause"]  = 2
        
        # set tumorsize attributes as integers
        dummy = ["0-4","5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"]
        for i in range(len(dummy)):
            data_set.loc[data_set["tumorSize"].astype(str) == dummy[i] , "tumorSize"] = i
            
        # set invNodes as integers
        dummy = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"]
        for i in range(len(dummy)):
            data_set.loc[data_set["invNodes"].astype(str) == dummy[i] , "invNodes"] = i
            
        # set nodeNodes as integers
        data_set.loc[data_set["nodeCaps"]=="yes", "nodeCaps"]  = 0
        data_set.loc[data_set["nodeCaps"]=="no",  "nodeCaps"]  = 1
        data_set.loc[data_set["nodeCaps"]=="?",   "nodeCaps"]  = 2
        
        # set breast integers
        data_set.loc[data_set["breast"]=="left",  "breast"]  = 0
        data_set.loc[data_set["breast"]=="right", "breast"]  = 1
        
        # set breastQuad as integers
        dummy = ["left_up", "left_low", "right_up", "right_low", "central", "?"]
        for i in range(len(dummy)):
            data_set.loc[data_set["breastQuad"] == dummy[i] , "breastQuad"] = i

        # set irradiat as integers
        data_set.loc[data_set["irradiat"]=="yes", "irradiat"]  = 0
        data_set.loc[data_set["irradiat"]=="no",  "irradiat"]  = 1

        Y = data_set.loc[:,"Class"]

        # Data scaling [0,1]
        data_set_final = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                      columns = list(data_set.columns))        
        X = data_set.drop("Class", axis=1)
        return X,Y

    def two_norm(self, data_set):
        #These data are already well-formatted
        #return data_set #(for tmva data prep)
        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    def german(self, data_set):
        data_set = data_set.copy()

        #title_mapping = {1: 1, 2: -1}
        data_set.loc[data_set["Class"] == 2, "Class"] = -1

        # set Status to integers
        dummy = ["A11", "A12", "A13", "A14"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Status"] == dummy[i] , "Status"] = i
            
        # set History to integers         
        dummy = ["A30", "A31", "A32", "A33", "A34"]
        for i in range(len(dummy)):
            data_set.loc[data_set["History"] == dummy[i] , "History"] = i

        # set Purpose to integers
        dummy = ["A40", "A41", "A42", "A43","A44","A45", "A46", "A47", "A48", "A49", "A410"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Purpose"] == dummy[i] , "Purpose"] = i

        # set Savings to integers
        dummy = ["A61", "A62", "A63", "A64", "A65"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Savings"] == dummy[i] , "Savings"] = i
            
        # set Employment to integers
        dummy = ["A71", "A72", "A73", "A74", "A75"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Employment"] == dummy[i] , "Employment"] = i
            
        # set SexStatus to integers
        dummy = ["A91", "A92", "A93", "A94", "A95"]
        for i in range(len(dummy)):
            data_set.loc[data_set["SexStatus"].astype(str) == dummy[i] , "SexStatus"] = i
        
        # set Debtor to integers
        dummy = ["A101", "A102", "A103"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Debtor"] == dummy[i] , "Debtor"] = i
                        
        dummy = ["A121", "A122", "A123", "A124"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Property"] == dummy[i] , "Property"] = i

        dummy = ["A141", "A142", "A143", "A144"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Plans"].astype(str) == dummy[i] , "Plans"] = i

        dummy = ["A151", "A152", "A153"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Housing"] == dummy[i] , "Housing"] = i

        dummy = ["A171", "A172", "A173",  "A174"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Job"] == dummy[i] , "Job"] = i

        dummy = ["A191", "A192"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Phone"] == dummy[i] , "Phone"] = i

        dummy = ["A201", "A202"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Foreign"] == dummy[i] , "Foreign"] = i
        
        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Class", axis=1)
        return X,Y

    # heart conditions data set
    def heart(self, data_set):
        data_set = data_set.copy()

        #dummy = [0: 1, 1: -1, 2: -1, 3: -1, 4: -1]
        dummy = [1, 2, 3, 4]
        for i in range(len(dummy)):
            data_set.loc[data_set["Class"] == dummy[i] , "Class"] = -1
        data_set.loc[data_set["Class"] == 0 , "Class"] = +1
        
        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Class", axis=1)
        return X,Y


    def solar(self, data_set):
        data_set = data_set.copy()
        
        dummy = ["A", "B", "C", "D", "E", "F", "H"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Zurich"].astype(str) == dummy[i] , "Zurich"] = i
    
        dummy = ["X", "R", "S", "A", "H", "K"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Size"].astype(str)  == dummy[i] , "Size"] = i
        
        dummy = ["X", "O", "I", "C"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Distro"].astype(str) == dummy[i] , "Distro"] = i
        
        #create the Class
        data_set["Class"]=data_set.sum(axis=1)
        data_set.loc[(data_set["Class1"] == 0) & (data_set["Class2"] == 0) & (data_set["Class3"] == 0), "Class"] = +1
        data_set.loc[(data_set["Class1"] != 0) | (data_set["Class2"] != 0) | (data_set["Class3"] != 0), "Class"] = -1
        data_set = data_set.drop(["Class1", "Class2", "Class3" ], axis=1)

        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Class", axis=1)
        return X,Y


    def car(self, data_set):
        data_set = data_set.copy()

        dummy = ["vhigh", "high", "med", "low"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Buy"].astype(str) == dummy[i] , "Buy"] = i
        
        dummy = ["vhigh", "high", "med", "low"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Maint"].astype(str) == dummy[i] , "Maint"] = i
            
        dummy = ["2", "3", "4", "5more"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Doors"].astype(str) == dummy[i] , "Doors"] = i
        
        dummy = ["2", "4", "more"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Persons"].astype(str) == dummy[i] , "Persons"] = i
        
        dummy = ["small", "med", "big"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Lug"].astype(str) == dummy[i] , "Lug"] = i
        
        dummy = ["low", "med", "high"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Safety"].astype(str) == dummy[i] , "Safety"] = i
        
        dummy = ["acc", "good", "vgood"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Class"].astype(str) == dummy[i] , "Class"] = 1
        data_set.loc[data_set["Class"].astype(str) == "unacc" , "Class"] = -1

        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Class", axis=1)
        return X,Y


    def ecoli(self, data_set):
        data_set = data_set.copy()

        dummy = ["im","pp","imU","om","omL","imL","imS"]
        for i in range(len(dummy)):
            data_set.loc[data_set["Class"]==dummy[i],  "Class"] = -1
        data_set.loc[data_set["Class"]=="cp",  "Class"] = 1
        
        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))
        X = data_set.drop("Class", axis=1)
        return X,Y
    
    def wine(self, data_set):
        data_set = data_set.copy()
        for i in range(11):
            if(i+1 <= 4): # wine quality, see wine.names
                data_set.loc[data_set["Class"] == i+1 , "Class"] = -1
            else:
                data_set.loc[data_set["Class"] == i+1 , "Class"] = +1
        # set class vector
        Y = data_set["Class"]
        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))        
        X = data_set.drop("Class", axis=1)
        return X,Y

    
    def abalone(self, data_set):
        data_set = data_set.copy()
        # set classes +-1
        for i in range(30):
            if(i+1 <= 11): # age restriction, see abalone.names
                data_set.loc[data_set["Class"].astype(str) == str(i+1), "Class"] = -1
            else:
                data_set.loc[data_set["Class"].astype(str) == str(i+1), "Class"] = +1
        # set sex atributes to integers
        data_set.loc[data_set["Sex"] == "M", "Sex"] = 0
        data_set.loc[data_set["Sex"] == "F", "Sex"] = 1
        data_set.loc[data_set["Sex"] == "I", "Sex"] = 2
        # set the class vector
        Y = data_set["Class"]
        # data normalization [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))        
        X = data_set.drop("Class", axis=1)
        return X,Y

    
    def adult(self, data_set):
        data_set = data_set.copy()
        # set sex atributes to integers
        data_set.loc[data_set["sex"] == "Male", "sex"] = 0
        data_set.loc[data_set["sex"] == "Female", "sex"] = 1
        # set workclass atributes to integers
        dummy = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]        
        for i in range(len(dummy)):
            data_set.loc[data_set["workclass"].astype(str) == dummy[i] , "workclass"] = i
        # set education atributes to integers
        dummy = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                 "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
        for i in range(len(dummy)):
            data_set.loc[data_set["education"].astype(str) == dummy[i] , "education"] = i
        # set marital status to integers
        dummy = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        for i in range(len(dummy)):
            data_set.loc[data_set["marital-status"].astype(str) == dummy[i] , "marital-status"] = i
        # set ocupation atributes to integers
        dummy = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                 "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        for i in range(len(dummy)):
            data_set.loc[data_set["occupation"].astype(str) == dummy[i] , "occupation"] = i
        # set relationship atributes to integers
        dummy = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        for i in range(len(dummy)):
            data_set.loc[data_set["relationship"].astype(str) == dummy[i] , "relationship"] = i
        # set race atributes to integers
        dummy = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        for i in range(len(dummy)):
            data_set.loc[data_set["race"].astype(str) == dummy[i] , "race"] = i            
        # set race atributes to integers
        dummy = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
                 "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                 "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
                 "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]        
        for i in range(len(dummy)):
            data_set.loc[data_set["native-country"].astype(str) == dummy[i] , "native-country"] = i
            
        # set the class vector
        Y = data_set["Class"]
        # data normalization [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))        
        X = data_set.drop("Class", axis=1)
        return X,Y

    def connect(self, data_set):
        data_set = data_set.copy()
        # set the class vector
        Y = data_set["Class"]
        # data normalization [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))        
        X = data_set.drop("Class", axis=1)
        return X,Y

        
    def nursery(self, data_set):

        title_mapping = {"usual": 0, "pretentious": 1, "great_pret": 2}
        data_set["Parents"] = data_set["Parents"].map(title_mapping)
        data_set["Parents"] = data_set["Parents"].fillna(0)

        title_mapping = {"proper" : 0, "less_proper" : 1, "improper" : 2, "critical" : 3, "very_crit" : 4}
        data_set["Has_nurs"] = data_set["Has_nurs"].map(title_mapping)
        data_set["Has_nurs"] = data_set["Has_nurs"].fillna(0)

        title_mapping = {"complete" : 0, "completed" : 1, "incomplete" : 2, "foster" : 3}
        data_set["Form_fam"] = data_set["Form_fam"].map(title_mapping)
        data_set["Form_fam"] = data_set["Form_fam"].fillna(0)

        title_mapping = {1: 0, 2: 1, 3: 2, "more": 4}
        data_set["Children"] = data_set["Children"].map(title_mapping)
        data_set["Children"] = data_set["Children"].fillna(0)

        title_mapping = {"convenient" : 0, "less_conv" : 1, "critical" : 2}
        data_set["Housing"] = data_set["Housing"].map(title_mapping)
        data_set["Housing"] = data_set["Housing"].fillna(0)

        title_mapping = {"convenient" : 0, "inconv" : 0}
        data_set["Finance"] = data_set["Finance"].map(title_mapping)
        data_set["Finance"] = data_set["Finance"].fillna(0)

        title_mapping = {"nonprob" : 0, "slightly_prob" : 1, "problematic" : 2}
        data_set["Social"] = data_set["Social"].map(title_mapping)
        data_set["Social"] = data_set["Social"].fillna(0)

        title_mapping = {"recommended" : 0, "priority" : 1, "not_recom" : 2}
        data_set["Health"] = data_set["Health"].map(title_mapping)
        data_set["Health"] = data_set["Health"].fillna(0)

        title_mapping = {"not_recom" : -1, "recommend" : 1, "very_recom" : 1, "priority" : 1, "spec_prior" : 1}
        data_set["Class"] = data_set["Class"].map(title_mapping)
        data_set["Class"] = data_set["Class"].fillna(0)

        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    def tac_toe(self, data_set):

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["TL"] = data_set["TL"].map(title_mapping)
        data_set["TL"] = data_set["TL"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["TM"] = data_set["TM"].map(title_mapping)
        data_set["TM"] = data_set["TM"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["TR"] = data_set["TR"].map(title_mapping)
        data_set["TR"] = data_set["TR"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["ML"] = data_set["ML"].map(title_mapping)
        data_set["ML"] = data_set["ML"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["MM"] = data_set["MM"].map(title_mapping)
        data_set["MM"] = data_set["MM"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["MR"] = data_set["MR"].map(title_mapping)
        data_set["MR"] = data_set["MR"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["BL"] = data_set["BL"].map(title_mapping)
        data_set["BL"] = data_set["BL"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["BM"] = data_set["BM"].map(title_mapping)
        data_set["BM"] = data_set["BM"].fillna(0)

        title_mapping = {"x": 0, "o": 1, "b": 2}
        data_set["BR"] = data_set["BR"].map(title_mapping)
        data_set["BR"] = data_set["BR"].fillna(0)

        title_mapping = {"positive" : 1, "negative" : -1}
        data_set["Class"] = data_set["Class"].map(title_mapping)
        data_set["Class"] = data_set["Class"].fillna(0)


        X = data_set.drop("Class", axis=1)
        Y = data_set["Class"]
        return X,Y


    # bin data?!?
    # Xin = data_set.drop("Class", axis=1)
    # est = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="uniform")
    # est.fit(Xin)
    
    # XT = est.transform(Xin)
    
    # # Creating pandas dataframe from numpy array
    # X = pd.DataFrame({"D0_m": XT[:, 0], "D0_p": XT[:, 1], "p0_p": XT[:, 2]})
