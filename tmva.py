#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

# tmva module

import ROOT
from ROOT import TFile, TMVA, TCut

m_outputFile = TFile( "TMVA.root", 'RECREATE' )
m_inputFile  = TFile.Open("./data/cancer.root" )
TMVA.Tools.Instance()

m_factory = TMVA.Factory( "TMVAClassification", m_outputFile #this is optional
                       ,"!V:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" )

m_loader = TMVA.DataLoader("dataset")

m_loader.AddVariable("age", 'F')
m_loader.AddVariable("menopause", 'F')
m_loader.AddVariable("tumorSize", 'F')
m_loader.AddVariable("invNodes", 'F')
m_loader.AddVariable("degMalig", 'F')
# m_loader.AddVariable("nodeCaps", 'F')
# m_loader.AddVariable("breast", 'I')
# m_loader.AddVariable("breastQuad")
# m_loader.AddVariable("irradiat")


# Get the signal and background trees for training
signal      = m_inputFile.Get("signal")
background  = m_inputFile.Get("background")

# Global event weights (see below for setting event-wise weights)
signalWeight     = 1.0
backgroundWeight = 1.0

m_loader.AddSignalTree(signal, signalWeight)
m_loader.AddBackgroundTree(background, backgroundWeight)

m_loader.fSignalWeight = signalWeight
m_loader.fBackgroundWeight = backgroundWeight
m_loader.fTreeS = signal
m_loader.fTreeB = background

mycuts = TCut("")
mycutb = TCut("")

m_loader.PrepareTrainingAndTestTree(mycuts, mycutb,
				    "nTrain_Signal=150:nTrain_Background=60:nTest_Signal=51:nTest_Background=25:SplitMode=Random:NormMode=NumEvents:!V" )

#BDT
m_factory.BookMethod(m_loader,TMVA.Types.kBDT, "BDT",
  		     "!V:NTrees=100:MinNodeSize=5%:MaxDepth=4:BoostType=AdaBoost:AdaBoostBeta=0.15:SeparationType=GiniIndex:nCuts=100" )

#Support Vector Machine
m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_RBF", "Kernel=RBF:Gamma=0.25:Tol=0.001:VarTransform=Norm" )
m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_Poly","Kernel=Polynomial:Theta=0.1:Order=2:Tol=0.001:VarTransform=Norm" )#Polynomial kernel
m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_prod_polRBF","Kernel=Prod:KernelList=RBF*Polynomial:Gamma=0.1:Theta=1:Order=1:Tol=0.001:VarTransform=Norm")#prod
m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_sum_polRBF", "Kernel=Sum:KernelList=RBF+Polynomial:Gamma=0.1:Theta=1:Order=1:Tol=0.001:VarTransform=Norm")#sum
# m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_MG",  "Kernel=MultiGauss:GammaList=0.25,0.25,0.25,0.25:Tol=0.001:VarTransform=Norm" )#MultiGauss kernel
# m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_prod_mgRBF", "Kernel=Prod:KernelList=RBF*MultiGauss:Gamma=0.25:GammaList=0.1,0.2:Tol=0.001:VarTransform=Norm")#prod
# m_factory.BookMethod( m_loader, TMVA.Types.kSVM, "SVM_sum_mgRBF",  "Kernel=Sum:KernelList=RBF+MultiGauss:Gamma=0.25:GammaList=0.1,0.2:Tol=0.001:VarTransform=Norm")#prod

#Train methods
m_factory.TrainAllMethods()
#Test and evaluate the model(s)
m_factory.TestAllMethods()
m_factory.EvaluateAllMethods()




# m_loader.AddVariable("Status")
# m_loader.AddVariable("Duration")
# m_loader.AddVariable("History")
# m_loader.AddVariable("Purpose")
# m_loader.AddVariable("Amount")
# m_loader.AddVariable("Savings")
# m_loader.AddVariable("Employment")
# m_loader.AddVariable("Income")
# m_loader.AddVariable("SexStatus")
# m_loader.AddVariable("Debtor")
# m_loader.AddVariable("Since")
# m_loader.AddVariable("Property")
# m_loader.AddVariable("Age")
# m_loader.AddVariable("Plans")
# m_loader.AddVariable("Housing")
# m_loader.AddVariable("Accounts")
# m_loader.AddVariable("Job")
# m_loader.AddVariable("People")
# m_loader.AddVariable("Phone")
# m_loader.AddVariable("Foreign")
