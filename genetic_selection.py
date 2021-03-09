#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
import numpy as np
import random
from random import randint
from collections import Counter
from tqdm import tqdm
from functools import lru_cache
import datetime
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,auc

# framework includes
from boostedSVM import AdaBoostSVM
from data_preparation import data_preparation
import data_utils as du


# Genetic algorithm for training sub-dataset selection
class genetic_selection:

    def __init__(self, model, isAB_SVM, X_train, Y_train, X_test, Y_test, pop_size, chrom_len, n_gen, coef, mut_rate):
        self.model = model
        self.AB_SVM = isAB_SVM
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test
        self.population_size=pop_size
        self.chrom_len = chrom_len
        self.n_generations = n_gen # maximum number of iterations
        self.coef = coef
        self.mutation_rate=mut_rate


    def execute(self): # run_genetic_algorithm(X_train, Y_train, population_size=5, chrom_len=100, n_generations=10, coef=0.5)
        #execute(X, y, population_size=10, chrom_len=10, n_generations=20, coef=0.5, mutation_rate=0.3)
        best_chromo = np.array([])
        best_score  = np.array([])

        next_generation_x, next_generation_y, next_generation_indexes = self.initialize_population(self.X_train, self.Y_train,
                                                                                                   self.population_size, self.chrom_len)
        
        for generation in tqdm(range(self.n_generations)):
            #print(np.unique(next_generation_y))
            scores, popx, popy, index = self.fitness_score(next_generation_x, next_generation_y, next_generation_indexes)
            scores, popx, popy, index = self.set_population_size(scores, popx, popy, index, generation, self.population_size)
            if self.termination_criterion(generation, best_score=scores, window=10):
                print('End of genetic algorithm')
                self.best_pop = next_generation_indexes
                break
            else:
                pa_x, pa_y, pb_x, pb_y, ind_a, ind_b = self.selection(popx, popy, index, self.coef)
                new_population_x , new_population_y, new_index = self.crossover(pa_x, pa_y, pb_x, pb_y, ind_a, ind_b, self.population_size)
                new_offspring_x, new_offspring_y, new_offs_index = self.mutation(new_population_x, new_population_y, new_index, self.mutation_rate)
                next_generation_x, next_generation_y, next_generation_indexes = self.append_offspring(popx, popy, new_offspring_x, new_offspring_y,
                                                                                                        index, new_offs_index)

            print(f"Best score achieved in generation {generation} is {scores[0]}")

        self.best_pop = next_generation_indexes
        
        
    def best_population(self):
        ''' fetches the best trained indexes, removing repetitions'''
        best_train_indexes = self.best_pop.flatten()
        return np.unique(best_train_indexes)

    
    def initialize_population(self, X, y, size, chromosome_length): # size==pop_size
        population_x, population_y, index_pop = [], [], []

        for i in range(size):
            chromosome_x, chromosome_y = self.get_subset(X, y, size=chromosome_length)
            population_x.append(chromosome_x.values)
            population_y.append(chromosome_y.values)
            # keep track of the indexes and propagate them during the GA selection
            index_pop.append(chromosome_x.index)

        return np.array(population_x),np.array(population_y),np.array(index_pop)

            
    def get_subset(self, X, y, size): #size==chrom_size
        # separate indices by class
        y0_index = y[y == -1].index
        y1_index = y[y ==  1].index

        # select a random subset of indexes of length size/2
        random_y0 = np.random.choice(y0_index, int(size/2), replace = False)
        random_y1 = np.random.choice(y1_index, int(size/2), replace = False)

        # concatenate indexes for balanced dataframes
        indexes = np.concatenate([random_y0, random_y1])

        # construct balanced datasets
        X_balanced = X.loc[indexes]
        y_balanced = y.loc[indexes]

        # delete useless variables
        del y0_index
        del y1_index

        # return shuffled dataframes
        rand_st  = randint(0, 10)
        
        return X_balanced.sample(frac=1, random_state=rand_st), y_balanced.sample(frac=1, random_state=rand_st) # checar random_state


    @lru_cache(maxsize = 1000)
    def memoization_score(self, tuple_chrom_x , tuple_chrom_y):
        chromosome_x, chromosome_y = np.asarray(tuple_chrom_x), np.asarray(tuple_chrom_y)
        self.model.fit(chromosome_x, chromosome_y[0])
        predictions = self.model.predict(self.X_test)
        acc_score      = accuracy_score(self.Y_test, predictions)
        return acc_score


    def fitness_score(self, pop_x, pop_y, indexes_pop):
        scores = np.array([])
        for chromosome_x, chromosome_y in zip(pop_x, pop_y):
            array_tuple_x = map(tuple, chromosome_x)
            array_tuple_y = map(tuple, chromosome_y.reshape((1, len(chromosome_y))))
            tuple_tuple_x = tuple(array_tuple_x)
            tuple_tuple_y = tuple(array_tuple_y)
            acc_score     = self.memoization_score(tuple_tuple_x , tuple_tuple_y)
            scores        = np.append(scores, acc_score)
            #print('Final test prediction:   ', accuracy_score(self.Y_test, predictions), len(self.Y_test), len(predictions))
            #area = self.area_roc(self.model, self.X_test, self.Y_test)
            if self.AB_SVM:  self.model.clean() # needed for AdaBoostSVM
            
        sorted_indexes  = np.argsort(-1*scores) # indexes sorted by score, see the cross check!
        return scores[sorted_indexes], pop_x[sorted_indexes], pop_y[sorted_indexes], indexes_pop[sorted_indexes]


    def set_population_size(self, scores, popx, popy, index, generation, size):
        '''Gets rid of lower part of population, restoring original size'''
        if generation == 0:
            pass
        else:
            scores = scores[:size]
            popx   = popx[:size]
            popy   = popy[:size]
            index   = index[:size]
        return scores, popx, popy, index


    def selection(self, pop_x, pop_y, data_index, coef):
        '''High-Low-fit selection'''

        # high fit and low fit parts of population
        indices = np.array([i for i in range(len(pop_x))])
        hf_indexes = indices[:int(len(indices)*coef)]
        lf_indexes = indices[int(len(indices)*coef):]

        hf = np.random.choice(hf_indexes, 1, replace=False)
        lf = np.random.choice(lf_indexes, 1, replace=False)

        pa_x = pop_x[hf]
        pa_y = pop_y[hf]

        in_a = data_index[hf]

        pb_x = pop_x[lf]
        pb_y = pop_y[lf]

        in_b = data_index[lf]

        return pa_x, pa_y, pb_x, pb_y, in_a, in_b


    def crossover(self, parent_a_x, parent_a_y, parent_b_x, parent_b_y, index_a, index_b, num_children):
        offspring_x = []
        offspring_y = []        
        offspring_index = []

        for i in range(0, num_children):
            p_ab_x = np.array([])
            p_ab_y = np.array([])

            i_ab = np.array([])

            # generate random indices
            rand_indx = np.random.choice(range(0,2*len(parent_a_x[0])), len(parent_a_x[0]), replace=False)

            p_ab_x = np.concatenate((parent_a_x[0], parent_b_x[0]), axis=0)
            p_ab_y = np.concatenate((parent_a_y[0], parent_b_y[0]), axis=0)

            i_ab = np.concatenate((index_a[0], index_b[0]), axis=0)

            new_x = p_ab_x[rand_indx]
            new_y = p_ab_y[rand_indx]

            new_i = i_ab[rand_indx]

            offspring_x.append(new_x)
            offspring_y.append(new_y)
            
            offspring_index.append(new_i)

        return np.array(offspring_x), np.array(offspring_y), np.array(offspring_index)
        #return new_x, new_y


    def mutation(self, offspring_x, offspring_y, index, mutation_rate):
        pop_nextgen_x = []
        pop_nextgen_y = []
        
        ind_nextgen = []

        for i in range(0, len(offspring_x)):
            chromosome_x = offspring_x[i]
            chromosome_y = offspring_y[i]

            index_chromosome = index[i]

            for j in range(len(chromosome_x)):
                if random.random() < mutation_rate:
                    while True:
                        # get random sample from X_train, Y_train
                        rand_st  = randint(0, 10)
                        random_x = X_train.sample(random_state=rand_st)
                        random_y = Y_train.sample(random_state=rand_st)
                        random_index = random_x.index

                        # Check if new random chromosome is already in the population. If not, it is added
                        if (chromosome_x == random_x.to_numpy()).all(1).any() is not True:
                            chromosome_x[j] = random_x.to_numpy()
                            chromosome_y[j] = random_y.to_numpy()
                            index_chromosome[j] = random_index.to_numpy()
                            break

            pop_nextgen_x.append(chromosome_x)
            pop_nextgen_y.append(chromosome_y)
            ind_nextgen.append(index_chromosome)

        return np.array(pop_nextgen_x), np.array(pop_nextgen_y), np.array(ind_nextgen) # check existence of genes -1 and 1 in Y, to avoid sklearn crashes


    def append_offspring(self, next_generation_x, next_generation_y, new_offspring_x, new_offspring_y, next_generation_indexes, new_off_index):
        '''Append offspring to population'''
        next_generation_x = np.append(next_generation_x, new_offspring_x, axis=0)
        next_generation_y = np.append(next_generation_y, new_offspring_y, axis=0)
        next_generation_indexes = np.append(next_generation_indexes, new_off_index, axis=0)

        return  next_generation_x, next_generation_y, next_generation_indexes


    def termination_criterion(self, generation, best_score, window=10):
        if generation <= window - 1:
            return False
        else:
            std = pd.Series(best_score).rolling(window).std() # equivalent to np.std(best_score, ddof=1)
            print(std, type(std), len(std), np.std(best_score,ddof=1), best_score)
            print('STD: ', std.iloc[len(std)-1])
            if std.iloc[len(std)-1] < 0.01:
                print('TRUE')
                return True
            else:
                return False

    def area_roc(self, model, X_test, Y_test):
        y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
        TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
        return auc(FPR,TPR)


# Experiments
sample_list = ['titanic', 'cancer', 'german', 'heart', 'solar','car','contra','tac_toe', 'belle2_i', 'belle2_ii','belle_iii']
data = data_preparation(GA_selection = True)
X_train, Y_train, X_test, Y_test = data.dataset('belle2_iii','',sampling=False,split_sample=0.4, train_test=True)
# X_train, Y_train, X_test, Y_test = data.dataset('titanic','',sampling=False,split_sample=0.4, train_test=False)
model_test = AdaBoostSVM(C=50, gammaIni=5, myKernel='rbf', Diversity=True, debug=False)
#model_test = SVC()

# Start training
start = datetime.datetime.now()
#model_test.fit(X_train,Y_train)
test_gen = genetic_selection(model_test, True, X_train, Y_train, X_test, Y_test,
                             pop_size=10, chrom_len=100, n_gen=1000, coef=0.5, mut_rate=0.3)
test_gen.execute()

best_train_indexes = test_gen.best_population()
print(best_train_indexes)
print(best_train_indexes.shape,  type(best_train_indexes))

end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed training time = " + str(elapsed_time))
