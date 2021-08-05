'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# model caller module
import numpy as np

# support vector machines
from boostedSVM import AdaBoostSVM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# n-word classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# probabilistic
from sklearn.naive_bayes      import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

# linear models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


def adaboost_svm(div_flag=False, my_c=150, my_gamma_end=100, myKernel='rbf', myDegree=1, myCoef0=1, early_stop=True, debug=False):
    # boosted support vector machine (ensemble)
    svmb = AdaBoostSVM(C=my_c, gammaEnd=my_gamma_end, myKernel=myKernel, myDegree=myDegree, myCoef0=myCoef0,
                       Diversity=div_flag, early_stop=early_stop, debug=debug)
    return svmb

def single_svm():
    # support vector machine (single case)
    return SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001)

def linear_svm():
    # support vector machine (linear case)
    # decision
    return LinearSVC()

def bdt_svm():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(base_estimator=SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001),
                              n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=None)

def bag_svm():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier(base_estimator=SVC(C=150.0, kernel='rbf', gamma=1/(2*(10**2)), shrinking = True, probability = True, tol = 0.001))

def rand_forest():
    # RANDOM forest classifier
    return RandomForestClassifier(n_estimators=100)

def bdt_forest():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(n_estimators=100, random_state=None)

def bag_forest():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier()

def grad_forest():
    # gradient boost classifier tree, this only for trees!
    return GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=None)
    
def neural_net():
    # Neural Network Multi Layer Perceptron classifier
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)

def k_neighbors():
    # K neighbors classifier. n_neighbors=3 because there are 2 classes
    return KNeighborsClassifier(n_neighbors=3)

def linear_dis():
    # to-do set values
    return LinearDiscriminantAnalysis()

def quad_dis():
    # to-do set values
    return QuadraticDiscriminantAnalysis()

def gauss_nb():
    # to-do: set values    
    return GaussianNB()

def gauss_pc():
    # to-do: set values
    return GaussianProcessClassifier()

def log_reg():
    # to-do: set values
    return LogisticRegression()

def ridge_class():
    # to-do: set values
    # decision
    return RidgeClassifier()

def sgdc_class():
    # to-do: set values
    # decision
    return SGDClassifier()

def pass_agre():
    # to-do: set values
    # decision
    return PassiveAggressiveClassifier()

def model_loader_batch(process):
    # return a single model to be used in a batch job
    batch_models = model_flavors()
    return (batch_models, batch_models[process])

def model_flavors():
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    models_auc = []
    mut_rate = 0.5
    models_auc.append(("trad-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=1, myCoef0=-1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))

    models_auc.append(("trad-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-sig-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    
    models_auc.append(("genHLAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))

    models_auc.append(("genHLAUC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    
    models_auc.append(("genHLACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))

    models_auc.append(("genHLACC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    
    
    models_auc.append(("genRLTAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))

    models_auc.append(("genRLTAUC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))

    models_auc.append(("genRLTACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))

    models_auc.append(("genRLTACC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=1000, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 100, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    
    return models_auc

    
def model_loader(option=None, sample_name=None):
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    # write a line to get model flavours

    models_auc = []

    # if option == "boot" or option=="kfold":        
    #     models_auc.append((adaboost_svm(False), "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(True),  "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(False), "boost-svm", "absv",  "trad"))
    #     models_auc.append((adaboost_svm(True),  "boost-svm", "absv",  "trad"))
    # elif option == "no_div":                                
    #     models_auc.append((adaboost_svm(True),  "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(True),  "boost-svm", "absv",  "trad"))
    # elif option == "diverse":                               
    #     models_auc.append((adaboost_svm(False), "b-svm-gen", "absv",  "gene"))
    #     models_auc.append((adaboost_svm(False), "boost-svm", "absv",  "trad"))


    #if option == "b2b":
    
    # models_auc.append((single_svm(),  "single-svm", "prob", "single-svm", "trad"))
    # models_auc.append((linear_svm(),  "linear-svm", "deci", "linear-svm", "trad"))
    # models_auc.append((bdt_svm(),     "bdt-svm",    "prob", "bdt-svm",    "trad"))
    # models_auc.append((bag_svm(),     "bag-svm",    "prob", "bag-svm",    "trad"))
    # models_auc.append((rand_forest(), "rand-forest","prob", "rand-forest","trad"))
    # models_auc.append((bdt_forest(),  "bdt-forest", "prob", "bdt-forest", "trad"))
    # models_auc.append((bag_forest(),  "bag-forest", "prob", "bag-forest", "trad"))
    # models_auc.append((grad_forest(), "grad-forest","prob", "grad-forest","trad"))
    # models_auc.append((neural_net(),  "neural-net", "prob", "neural-net", "trad"))
    # models_auc.append((k_neighbors(), "k-neigh",    "prob", "k-neigh",    "trad"))
    # models_auc.append((gauss_nb(),    "gauss-nb",   "prob", "gauss-nb",   "trad"))
    # models_auc.append((gauss_pc(),    "gauss-pc",   "prob", "gauss-pc",   "trad"))
    # models_auc.append((log_reg(),     "log-reg",    "prob", "log-reg",    "trad"))
    # models_auc.append((ridge_class(), "ridge-cl",   "deci", "ridge-cl",   "trad"))
    # models_auc.append((sgdc_class(),  "sgdc-cl",    "deci", "sgdc-cl",    "trad"))
    # models_auc.append((pass_agre(),   "pass-agre",  "deci", "pass-agre",  "trad"))
    # if sample_name != "belle2_ii" and sample_name != "solar": # ugly fix
    #     models_auc.append((linear_dis(),  "prob", "linear-dis", "trad"))
    #     models_auc.append((quad_dis(),    "prob", "quad-dis", "trad"))

    return models_auc


    # models_auc.append(("genHLPREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    # models_auc.append(("genHLPREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "prec", "highlow", 0.5))
    
    # models_auc.append(("genHLF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    
    # models_auc.append(("genHLF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    # models_auc.append(("genHLF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "highlow", 0.5))
    
    # models_auc.append(("genHLREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    
    # models_auc.append(("genHLREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    # models_auc.append(("genHLREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "highlow", 0.5))
    
    # models_auc.append(("genHLGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    
    # models_auc.append(("genHLGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))
    # models_auc.append(("genHLGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "highlow", 0.5))


        # models_auc.append(("genRLTF1-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    
    # models_auc.append(("genRLTF1-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    # models_auc.append(("genRLTF1-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "f1", "roulette", 0.5))
    
    # models_auc.append(("genRLTREC-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    
    # models_auc.append(("genRLTREC-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    # models_auc.append(("genRLTREC-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "rec", "roulette", 0.5))
    
    # models_auc.append(("genRLTGMN-lin-NOTdiv", adaboost_svm(div_flag=False, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-rbf-NOTdiv", adaboost_svm(div_flag=False, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-pol-NOTdiv", adaboost_svm(div_flag=False, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-sig-NOTdiv", adaboost_svm(div_flag=False, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    
    # models_auc.append(("genRLTGMN-lin-YESdiv", adaboost_svm(div_flag=True, myKernel='linear',  myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-rbf-YESdiv", adaboost_svm(div_flag=True, myKernel='rbf',     myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-pol-YESdiv", adaboost_svm(div_flag=True, myKernel='poly',    myDegree=2, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
    # models_auc.append(("genRLTGMN-sig-YESdiv", adaboost_svm(div_flag=True, myKernel='sigmoid', myDegree=1, myCoef0=1), "absv",  "gene", 0.3, "gmean", "roulette", 0.5))
