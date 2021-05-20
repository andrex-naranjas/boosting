'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

from os import system,popen
import os

# create directories
system('mkdir plots tables')
system('mkdir output && cd output && mkdir abalone  adult  belle2_i  belle2_ii  belle2_iii  belle2_iv  belle_iii  cancer  car  connect  contra  ecoli  german  heart  solar  tac_toe  titanic  wine && cd ..')
system('mkdir stats_results && cd stats_results && mkdir abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')
system('mkdir output_batch && cd output_batch && mkdir abalone  adult  cancer  car  connect  ecoli  german  heart  solar  titanic  wine && cd ..')

# set enviroment variables for python
PYTHONPATH = popen('which python3').read().strip()
BIN = "/bin/python3"

if BIN in PYTHONPATH:
    PYTHONHOME = PYTHONPATH.replace(BIN,'')
# system('export PYTHONPATH="{0}"'.format(PYTHONPATH))
# system('export PYTHONHOME="{0}"'.format(PYTHONHOME))
os.environ['PYTHONPATH'] = str(PYTHONPATH)
os.environ['PYTHONHOME'] = str(PYTHONHOME)

print('Please manually set the python enviroments (this should be fixed in the future):')
print('export PYTHONPATH="{0}"'.format(PYTHONPATH))
print('export PYTHONHOME="{0}"'.format(PYTHONHOME))
# export PYTHONPATH="/u/user/andres/.conda/envs/flavour/bin/python3"
# export PYTHONHOME="/u/user/andres/.conda/envs/flavour"
