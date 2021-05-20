'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''

from os import system,popen
import os

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
