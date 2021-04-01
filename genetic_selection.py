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
from tqdm import tqdm
from functools import lru_cache
import datetime
import pandas as pd
from sklearn.utils import resample

from sklearn.metrics import accuracy_score,auc,precision_score,roc_auc_score,f1_score,recall_score

# framework includes
from data_preparation import data_preparation
import data_utils as du


# Genetic algorithm for training sub-dataset selection
class genetic_selection:

    def __init__(self, model, model_type, X_train, Y_train, X_test, Y_test, pop_size, chrom_len, n_gen, coef, mut_rate, score_type='acc'):
        self.model = model
        self.model_type = model_type
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test
        self.population_size=pop_size
        self.chrom_len = chrom_len
        self.n_generations = n_gen # maximum number of iterations
        self.coef = coef
        self.mutation_rate=mut_rate
        self.score_type = score_type
        if(model_type == 'absv'):
            self.AB_SVM = True
        else:
            self.AB_SVM = False

    def execute(self):
        best_chromo = np.array([])
        best_score  = []
        next_generation_x, next_generation_y, next_generation_indexes = self.initialize_population(self.X_train, self.Y_train,
                                                                                                   self.population_size, self.chrom_len)
        for generation in tqdm(range(self.n_generations)):
            #print(np.unique(next_generation_y))
            scores, popx, popy, index = self.fitness_score(next_generation_x, next_generation_y, next_generation_indexes)
            scores, popx, popy, index = self.set_population_size(scores, popx, popy, index, generation, self.population_size)
            if self.termination_criterion(generation, best_score, window=10):
                print('End of genetic algorithm')
                break
            else:
                pa_x, pa_y, pb_x, pb_y, ind_a, ind_b = self.selection(popx, popy, index, self.coef)
                new_population_x , new_population_y, new_index = self.crossover(pa_x, pa_y, pb_x, pb_y, ind_a, ind_b, self.population_size)
                new_offspring_x, new_offspring_y, new_offs_index = self.mutation(new_population_x, new_population_y, new_index, self.mutation_rate)
                best_score.append(scores[0])                
                next_generation_x, next_generation_y, next_generation_indexes = self.append_offspring(popx, popy, new_offspring_x, new_offspring_y,
                                                                                                        index, new_offs_index)
            print(f"Best score achieved in generation {generation} is {scores[0]}")

        self.best_pop = index
        
        
    def best_population(self):
        ''' fetches the best trained indexes, removing repetitions'''
        best_train_indexes = self.best_pop.flatten()
        return np.unique(best_train_indexes)

    
    def initialize_population(self, X, y, size, chromosome_length): # size==pop_size
        population_x, population_y, index_pop = [], [], []

        for i in range(size):
            chromosome_x, chromosome_y = self.get_subset(X, y, size=chromosome_length, count=i)
            population_x.append(chromosome_x.values)
            population_y.append(chromosome_y.values)
            # keep track of the indexes and propagate them during the GA selection
            index_pop.append(chromosome_x.index)

        return np.array(population_x),np.array(population_y),np.array(index_pop)

            
    def get_subset(self, X, y, size, count): #size==chrom_size
        '''construct chromosomes'''        
        # separate indices by class
        y0_index = y[y == -1].index
        y1_index = y[y ==  1].index

        # set the size, to prevent larger sizes than allowed
        if(len(y0_index) < size/2 or len(y1_index) < size/2):
            size = np.amin([len(y0_index), len(y1_index)])
        
        # select a random subset of indexes of length size/2
        random_y0 = np.random.choice(y0_index, int(size/2), replace = False)
        random_y1 = np.random.choice(y1_index, int(size/2), replace = False)

        # concatenate indexes for balanced dataframes
        indexes = np.concatenate([random_y0, random_y1])

        # construct balanced datasets
        X_balanced = X.loc[indexes]
        y_balanced = y.loc[indexes]

        # shuffled dataframes
        rand_st = randint(0, 10)
        X_balanced = X_balanced.sample(frac=1, random_state=rand_st)
        y_balanced = y_balanced.sample(frac=1, random_state=rand_st) # check random_state
        
        # It may exist repeated indexes that change the chromosome size
        # these lines fix the issue coming from bootstrap
        # the GA selection cannot handle different chromosome sizes
        if(len(X_balanced) != len(indexes)):
            X_balanced = resample(X_balanced, replace=False, n_samples=len(indexes))
            y_balanced = resample(y_balanced, replace=False, n_samples=len(indexes))
            
        return X_balanced, y_balanced
    

    @lru_cache(maxsize = 1000)
    def memoization_score(self, tuple_chrom_x , tuple_chrom_y):
        chromosome_x, chromosome_y = np.asarray(tuple_chrom_x), np.asarray(tuple_chrom_y)
        self.model.fit(chromosome_x, chromosome_y[0])
        predictions = self.model_predictions(self.X_test, self.model_type, self.score_type)
        score       = self.score_value(self.Y_test, predictions, self.model_type, self.score_type)
        return score


    def fitness_score(self, pop_x, pop_y, indexes_pop):
        scores = np.array([])
        for chromosome_x, chromosome_y in zip(pop_x, pop_y):
            array_tuple_x = map(tuple, chromosome_x)
            array_tuple_y = map(tuple, chromosome_y.reshape((1, len(chromosome_y))))
            tuple_tuple_x = tuple(array_tuple_x)
            tuple_tuple_y = tuple(array_tuple_y)
            score         = self.memoization_score(tuple_tuple_x , tuple_tuple_y)
            scores        = np.append(scores, score)
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
            index  = index[:size]
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
                        random_x = self.X_train.sample(random_state=rand_st)
                        random_y = self.Y_train.sample(random_state=rand_st)
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
            if std.iloc[len(std)-1] < 0.01:
                return True
            else:
                return False

            
    def score_value(self, Y_test, y_pred, model_type, score_type):
        '''Computes different scores given options'''
        if(score_type == 'auc' and model_type == 'absv'):
            TPR, FPR = du.roc_curve_adaboost(y_pred, Y_test)
            score_value = auc(FPR,TPR)
        elif(score_type == 'auc' and model_type != 'absv'):
            score_value = roc_auc_score(Y_test, y_pred)    
        elif(score_type == 'acc'):
            score_value = accuracy_score(Y_test, y_pred)
        elif(score_type == 'prec'):
            score_value = precision_score(Y_test, y_pred)
        elif(score_type == 'f1'):
            score_value = f1_score(Y_test, y_pred)
        elif(score_type == 'rec'):
            score_value = recall_score(Y_test, y_pred)
        elif(score_type == 'gmean'):
            score_value  = np.sqrt(precision_score(Y_test, y_pred)*recallscore(Y_test, y_pred))
        return score_value

    def model_predictions(self, X_test, model_type, score_type):
        ''' computes the prediction given the score type set '''
        if(score_type == 'auc'):
            if(model_type == 'absv'):
                return self.model.decision_thresholds(X_test, glob_dec=True)
            elif(model_type == 'prob'):
                return self.model.predict_proba(X_test)[:,1]
            elif(model_type == 'deci'):
                return self.model.decision_function(X_test)
        else:
            return self.model.predict(X_test)
