
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# htcondor job submitter
# python3 submit_stats_svm.py sample_name boot/kfold

from sys import argv
from os import system,getenv,getuid,getcwd,popen
import model_maker as mm

workpath=getcwd()

if(len(argv)!=3):
  sample_name = 'titanic'
  boot_kfold = 'boot'
elif(len(argv)==3):
  sample_name = argv[1]
  boot_kfold = argv[2]

n_flavors = len(mm.model_flavors())
py3_path = popen('which python3').read().strip()

classad='''
universe = vanilla
executable = {0}
getenv = True
arguments = {1}/batch_stats_summary.py $(Process) {2} {1} {3}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
output = {1}/output_batch/{2}/$(Process).out
error = {1}/output_batch/{2}/$(Process).err
log = {1}/output_batch/{2}/$(Process).log
Queue {4}

'''.format(py3_path, workpath, sample_name, boot_kfold, n_flavors)

logpath = '.'

with open(logpath+'/condor.jdl','w') as jdlfile:
  jdlfile.write(classad)

print("************************ Batch jobs for: ", sample_name, "and stats:", boot_kfold, "************************")
#print('condor_submit %s/condor.jdl'%logpath)
system('condor_submit %s/condor.jdl'%logpath)
