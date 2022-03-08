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

#Train methods
m_factory.TrainAllMethods()
#Test and evaluate the model(s)
m_factory.TestAllMethods()
m_factory.EvaluateAllMethods()
