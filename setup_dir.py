''''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
from os import system

# create directories
system('mkdir -p plots tables')
system('mkdir -p output && cd output && mkdir -p abalone  adult  belle2_i  belle2_ii  belle2_iii  belle2_iv  belle_iii  cancer  car  connect  contra  ecoli  german  heart  solar  tac_toe  titanic  wine && cd ..')
system('mkdir -p stats_results && cd stats_results && mkdir -p abalone && cd abalone && mkdir -p boot kfold')
system('cd stats_results && mkdir -p adult && cd adult && mkdir -p boot kfold')
system('cd stats_results && mkdir -p cancer && cd cancer && mkdir -p boot kfold')
system('cd stats_results && mkdir -p car && cd car && mkdir -p boot kfold')
system('cd stats_results && mkdir -p connect && cd connect && mkdir -p boot kfold')
system('cd stats_results && mkdir -p ecoli && cd ecoli && mkdir -p boot kfold')
system('cd stats_results && mkdir -p german && cd german && mkdir -p boot kfold')
system('cd stats_results && mkdir -p heart && cd heart && mkdir -p boot kfold')
system('cd stats_results && mkdir -p solar && cd solar && mkdir -p boot kfold')
system('cd stats_results && mkdir -p titanic && cd titanic && mkdir -p boot kfold')
system('cd stats_results && mkdir -p wine && cd wine && mkdir -p boot kfold')
system('mkdir -p output_batch && cd output_batch && mkdir -p abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
