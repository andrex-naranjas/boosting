'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import data_visualization as dv
import data_utils as du


class model_performance:

    def __init__(self, model, X_train, Y_train, X_test, Y_test):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test            
        self.weights, self.errors, self.precision = model.get_metrics()


    def execute(self):
        test_pre = (self.model.predict(self.X_test) == self.Y_test).mean()
        test_err = (self.model.predict(self.X_test) != self.Y_test).mean()
        print(f'Test prec.: {test_pre:.1%}')
        print(f'Test error: {test_err:.1%}')

        # precision plot
        dv.plot_frame(pd.DataFrame(self.precision*100,np.arange(self.precision.shape[0])),
                      'Classifier precision', 'Classifier', 'training precision (%)', True, 0, 100,'belle2_iii')
        # errors plot
        dv.plot_frame(pd.DataFrame(self.errors*100,np.arange(self.errors.shape[0])),
                      'Classifier error', 'Classifier', 'training error (%)', True, 0, 100,'belle2_iii')
                
        # grid hyper parameter 2D-plots
        matrix = du.grid_param_gauss(self.X_train, self.Y_train, self.X_test, self.Y_test, sigmin=-5, sigmax=5, cmin=0, cmax=6)
        dv.plot_2dmap(matrix,-5,5,0,6,'belle2_iii')
        
        #boostrap error VS number of classiffiers calculation
        frame = du.error_number('titanic',myC=50,myGammaIni=10, train_test=False)
        classifier = "trad-rbf-YESdiv"
        classifier = "genHLACC-rbf-NOTdiv"
        classifier = "genHLAUC-sig-YESdiv"
        # classifier = "genHLACC-pol-YESdiv"        
        dv.plot_frame(frame, classifier, 'No. Classifiers', 'Test error (%)', False, 0, 50,'titanic')
        print(frame)
        frame.to_csv(classifier+"_number.csv", index=False)
        
