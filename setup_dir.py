'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
from os import system

# create directories
system('mkdir -p plots tables')
system('mkdir -p output && cd output && mkdir -p abalone  adult  belle2_i  belle2_ii  belle2_iii  belle2_iv  belle_iii  cancer  car  connect  contra  ecoli  german  heart  solar  tac_toe  titanic  wine && cd ..')
system('mkdir -p stats_results && cd stats_results && mkdir -p abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
system('mkdir -p output_batch && cd output_batch && mkdir -p abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
