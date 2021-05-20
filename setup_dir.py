'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
from os import system

# create directories
system('mkdir plots tables')
system('mkdir output && cd output && mkdir abalone  adult  belle2_i  belle2_ii  belle2_iii  belle2_iv  belle_iii  cancer  car  connect  contra  ecoli  german  heart  solar  tac_toe  titanic  wine && cd ..')
system('mkdir stats_results && cd stats_results && mkdir abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
system('mkdir output_batch && cd output_batch && mkdir abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
