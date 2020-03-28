#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

# tmva module

import ROOT
from ROOT import TFile, TMVA, TCut

outputFile = TFile( "TMVA.root", 'RECREATE' )
TMVA.Tools.Instance()

factory = TMVA.Factory( "TMVAClassification", outputFile #this is optional
                       ,"!V:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" )
