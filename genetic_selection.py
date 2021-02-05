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
import datetime

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# framework includes
from boostedSVM import AdaBoostSVM
from data_preparation import data_preparation



# Genetic algorithm for training sub-dataset selection
class genetic_selection:

    def __init__(self, model, X_train, Y_train, X_test, Y_test, pop_size, chrom_len, n_gen, coef, mut_rate):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test
        self.population_size=pop_size
        self.chrom_len = chrom_len
        self.n_generations = n_gen
        self.coef = coef
        self.mutation_rate=mut_rate 
        

    def execute(self):
    
        best_chromo = np.array([])
        best_score  = np.array([])
    
        next_generation_x, next_generation_y = self.initialize_population(X, y, size=population_size, chromosome_length=chrom_len)
    
    
        for generation in tqdm(range(n_generations)):
            #print(np.unique(next_generation_y))
            scores, popx, popy                   = fitness_score(next_generation_x, next_generation_y)
            scores, popx, popy                   = set_population_size(scores, popx, popy, generation, size=population_size)
            # termination_criterion()
            pa_x, pa_y, pb_x, pb_y               = selection(popx, popy, coef)
            new_population_x , new_population_y  = crossover(pa_x, pa_y, pb_x, pb_y, num_children=population_size)
            new_offspring_x, new_offspring_y     = mutation(new_population_x, new_population_y, mutation_rate=mutation_rate)
            next_generation_x, next_generation_y = append_offspring(next_generation_x, next_generation_y, new_offspring_x, new_offspring_y)
    
            print(f"Best score achieved in generation {generation} is {scores[-1::]}")
        return None

    
    def get_subset(self, X, y, size):
        # separate indices by class
        y0_index = y[y == -1].index
        y1_index = y[y == 1].index
            
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


    def initialize_population(self, X, y, size, chromosome_length):
        population_x, population_y = [], []
        for i in range(size):
            chromosome_x, chromosome_y = get_subset(X, y, size=chromosome_length)
            population_x.append(chromosome_x)
            population_y.append(chromosome_y)
        return np.array(population_x), np.array(population_y)


    def fitness_score(self, population_x, population_y):
        scores = np.array([])
        for chromosome_x, chromosome_y in zip(population_x, population_y):
            model.fit(chromosome_x, chromosome_y) # change to AdaBoostSVM
            predictions = model.predict(X_test)
            scores      = np.append(scores, accuracy_score(Y_test, predictions))

        sorted_indexes  = np.argsort(scores) # indexes sorted by score
        return scores[sorted_indexes], population_x[sorted_indexes], population_y[sorted_indexes]

    def set_population_size(scores, popx, popy, generation, size):
        '''Gets rid of lower part of population, restoring original size'''
        if generation == 0:
            pass
        else:
            scores = scores[size:]
            popx   = popx[size:]
            popy   = popy[size:]

        return scores, popx, popy

    def selection(self, pop_x, pop_y, coef):
        '''High-Low-fit selection'''
        
        # high fit and low fit parts of population
        indices = np.array([i for i in range(len(pop_x))])
        hf_indexes = indices[int(len(indices)*coef):]
        lf_indexes = indices[:int(len(indices)*coef)]

        hf = np.random.choice(hf_indexes, 1, replace=False)
        lf = np.random.choice(lf_indexes, 1, replace=False)

        pa_x = pop_x[hf]
        pa_y = pop_y[hf]

        pb_x = pop_x[lf]
        pb_y = pop_y[lf]
        
        return pa_x, pa_y, pb_x, pb_y
    

    def crossover(self, parent_a_x, parent_a_y, parent_b_x, parent_b_y, num_children):
        offspring_x = []
        offspring_y = []

        for i in range(0, num_children):
            p_ab_x = np.array([])
            p_ab_y = np.array([])

            # generate random indices
            rand_indx = np.random.choice(range(0,2*len(parent_a_x[0])), len(parent_a_x[0]), replace=False)

            p_ab_x = np.concatenate((parent_a_x[0], parent_b_x[0]), axis=0)
            p_ab_y = np.concatenate((parent_a_y[0], parent_b_y[0]), axis=0)

            new_x = p_ab_x[rand_indx]
            new_y = p_ab_y[rand_indx]

            offspring_x.append(new_x)
            offspring_y.append(new_y)
            
        return np.array(offspring_x), np.array(offspring_y)
        #return new_x, new_y
        

    def mutation(self, offspring_x, offspring_y, mutation_rate):
        pop_nextgen_x = []
        pop_nextgen_y = []
    
        for i in range(0, len(offspring_x)):
            chromosome_x = offspring_x[i]
            chromosome_y = offspring_y[i]
    
            for j in range(len(chromosome_x)):
                if random.random() < mutation_rate:
                    while True:
                        # get random sample from X_train, Y_train
                        rand_st  = randint(0, 10)
                        random_x = X_train.sample(random_state=rand_st)
                        random_y = Y_train.sample(random_state=rand_st)
    
                        # Check if new random chromosome is already in the population. If not, it is added
                        if (chromosome_x == random_x.to_numpy()).all(1).any() is not True:
                            chromosome_x[j] = random_x.to_numpy()
                            chromosome_y[j] = random_y.to_numpy()
                            break
    
            pop_nextgen_x.append(chromosome_x)
            pop_nextgen_y.append(chromosome_y)
    
        return np.array(pop_nextgen_x), np.array(pop_nextgen_y) # Asegurarnos que hayan genes -1 y 1 en Y, sino sklearn bastardea
    
    
    def append_offspring(self, next_generation_x, next_generation_y, new_offspring_x, new_offspring_y):
        '''Append offspring to population'''
        next_generation_x = np.append(next_generation_x, new_offspring_x, axis=0)
        next_generation_y = np.append(next_generation_y, new_offspring_y, axis=0)
    
        return  next_generation_x, next_generation_y
    
    def termination_criterion(self):
        if generation == 0:
            pass
        else:
            pass
            

    
# Experiments
data = data_preparation()
X_train, Y_train, X_test, Y_test = data.dataset('titanic','',sampling=False,split_sample=0.4)

model_test = AdaBoostSVM(C=50, gammaIni=10, myKernel='rbf')
#model_test = SVC()

X, y, population_size=10, chrom_len=10, n_generations=20, coef=0.5, mutation_rate=0.3

# Start training
start = datetime.datetime.now()

model_test.fit(X_train,Y_train)

end = datetime.datetime.now()
elapsed_time = end - start

print("Elapsed training time = " + str(elapsed_time))

# Start predicting
start = datetime.datetime.now()

predictions = model_test.predict(X_test)
print("Accuracy = "+ str(accuracy_score(Y_test,predictions)))
end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed fitting time = " + str(elapsed_time))

model = AdaBoostSVM(C=50, gammaIni=10, myKernel='rbf')

start = datetime.datetime.now()
#run_genetic_algorithm(X_train, Y_train, population_size=5, chrom_len=100, n_generations=10, coef=0.5)

end = datetime.datetime.now()
elapsed_time = end - start
print("Elapsed total time = " + str(elapsed_time))
