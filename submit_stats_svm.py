#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------
 Code to improve SVM
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
 ---------------------------------------------------------------
'''
# htcondor job submitter

from sys import argv
from os import system,getenv,getuid,getcwd

workpath=getcwd()

if(len(argv)!=3):
  sample_name = 'titanic'
  boot_kfold = 'boot'
elif(len(argv)==3):
  sample_name = argv[1]
  boot_kfold = argv[2]

classad='''
universe = vanilla
executable = /usr/bin/python3
arguments = {0}/batch_stats_summary.py $(Process) {1} {0} {2}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
output = {0}/output_batch/{1}/$(Process).out
error = {0}/output_batch/{1}/$(Process).err
log = {0}/output_batch/{1}/$(Process).log
Queue 1

'''.format(workpath, sample_name, boot_kfold)

logpath = '.'

with open(logpath+'/condor.jdl','w') as jdlfile:
  jdlfile.write(classad)

print("************************ Batch jobs for: ", sample_name, "and stats:", boot_kfold, "************************")
#print('condor_submit %s/condor.jdl'%logpath)
system('condor_submit %s/condor.jdl'%logpath)
