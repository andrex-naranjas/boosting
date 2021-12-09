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

def single_svm(my_kernel):
    # support vector machine (single case)
    my_C = 100
    my_gamma = 100
    my_coef = +1
    if my_kernel == 'sigmoid':
        my_coef = -1
        my_gamma = 10
    elif my_kernel == 'poly':
        my_C = 10
        my_gamma = 0.1        
            
    return SVC(C=my_C, kernel=my_kernel, degree=2, coef0=my_coef, gamma=10, shrinking = True, probability = True, tol = 0.001)

def linear_svm():
    # support vector machine (linear case)
    # decision
    return LinearSVC(C=10.0)

def bdt_svm():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(base_estimator=SVC(C=100.0, kernel='rbf', gamma=100, shrinking = True, probability = True, tol = 0.001),
                              n_estimators=100, learning_rate=1.0, algorithm='SAMME', random_state=None)

def bag_svm():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier(base_estimator=SVC(C=100.0, kernel='rbf', gamma=100, shrinking = True, probability = True, tol = 0.001))

def rand_forest():
    # RANDOM forest classifier
    return RandomForestClassifier(n_estimators=100)

def bdt_forest():
    # AdaBoost-SAMME: Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.
    return AdaBoostClassifier(n_estimators=100, random_state=1)

def bag_forest():
    # bagging (bootstrap) default base classifier, decision_tree
    return BaggingClassifier(n_estimators=100)

def grad_forest():
    # gradient boost classifier tree, this only for trees!
    return GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=1)
    
def neural_net():
    # Neural Network Multi Layer Perceptron classifier
    #return MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(2, 2), random_state=1)
    return MLPClassifier(solver='sgd', random_state=1)

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

def model_loader_batch(process, ensemble_single='ensemble'):
    # return a single model to be used in a batch job
    if ensemble_single =='ensemble':
        batch_models = model_flavors_ensemble()
    elif ensemble_single=='single':
        batch_models = model_flavors_single()
        
    return (batch_models, batch_models[process])

def model_flavors_ensemble():
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    models_auc = []
    mut_rate = 0.25
    models_auc.append(("trad-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=1, myCoef0=-1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))

    models_auc.append(("trad-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    models_auc.append(("trad-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "trad", mut_rate, "auc", "roulette", 0.0))
    
    models_auc.append(("genHLAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))

    models_auc.append(("genHLAUC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    models_auc.append(("genHLAUC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "highlow", 0.5))
    
    models_auc.append(("genHLACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))

    models_auc.append(("genHLACC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c=10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    models_auc.append(("genHLACC-lin-YESdiv", adaboost_svm(div_flag=True, my_c=10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "highlow", 0.5))
    
    
    models_auc.append(("genRLTAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))

    models_auc.append(("genRLTAUC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))
    models_auc.append(("genRLTAUC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "roulette", 0.5))

    models_auc.append(("genRLTACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))

    models_auc.append(("genRLTACC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))
    models_auc.append(("genRLTACC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "roulette", 0.5))

    
    models_auc.append(("genTNAUC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c=10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c=10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))

    models_auc.append(("genTNAUC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))
    models_auc.append(("genTNAUC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "auc", "tournament", 0.5))

    models_auc.append(("genTNACC-rbf-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-sig-NOTdiv", adaboost_svm(div_flag=False, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-pol-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-lin-NOTdiv", adaboost_svm(div_flag=False, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))

    models_auc.append(("genTNACC-rbf-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=100, myKernel='rbf',     myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-sig-YESdiv", adaboost_svm(div_flag=True, my_c=100, my_gamma_end=0.1, myKernel='sigmoid', myDegree=2, myCoef0=-1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-pol-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='poly',    myDegree=2, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))
    models_auc.append(("genTNACC-lin-YESdiv", adaboost_svm(div_flag=True, my_c= 10, my_gamma_end=0.1, myKernel='linear',  myDegree=1, myCoef0=+1), "absv",  "gene", mut_rate, "acc", "tournament", 0.5))

    return models_auc


def model_flavors_single():
    # set the models,their method to calculate the ROC(AUC), table name and selection
    # tuple = (model_latex_name, model, auc, selection, GA_mutation, GA_selection, GA_highLow_coef)

    models_auc = []
    mut_rate = 0.5
    # different models
    models_auc.append(("rbf-svm", single_svm("rbf"),        "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 0
    models_auc.append(("poly-svm", single_svm("poly"),      "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 1
    models_auc.append(("sigmoid-svm", single_svm("sigmoid"),"prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 2
    # models_auc.append(("linear-svm", linear_svm(),          "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 3
    
    # models_auc.append(("bdt-svm",    bdt_svm(),             "prob", "trad", mut_rate, "auc", "roulette", 0.0))
    
    models_auc.append(("bag-svm",    bag_svm(),             "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 4
    # models_auc.append(("rand-forest",rand_forest(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 5
    # models_auc.append(("bdt-forest", bdt_forest(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 6
    # models_auc.append(("bag-forest", bag_forest(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 7
    # models_auc.append(("grad-forest",grad_forest(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 8
    # models_auc.append(("neural-net", neural_net(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 9
    # models_auc.append(("k-neigh",    k_neighbors(),         "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 10
    # models_auc.append(("gauss-nb",   gauss_nb(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 11
    # models_auc.append(("gauss-pc",   gauss_pc(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 12
    # models_auc.append(("log-reg",    log_reg(),             "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 13
    # models_auc.append(("ridge-cl",   ridge_class(),         "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 14
    # models_auc.append(("sgdc-cl",    sgdc_class(),          "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 15
    # models_auc.append(("pass-agre",  pass_agre(),           "deci", "trad", mut_rate, "auc", "roulette", 0.0))  # 16
    # models_auc.append(("linear-dis", linear_dis(),          "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 17
    # models_auc.append(("quad-dis",   quad_dis(),            "prob", "trad", mut_rate, "auc", "roulette", 0.0))  # 18
    
    # if sample_name != "belle2_ii" and sample_name != "solar": # ugly fix
    #     models_auc.append((linear_dis(),  "prob", "linear-dis", "trad"))
    #     models_auc.append((quad_dis(),    "prob", "quad-dis", "trad"))
    
    return models_auc
    
