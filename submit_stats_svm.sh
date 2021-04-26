#sample_name = name
Universe = vanilla
Executable = /usr/bin/python3
Arguments = batch_stats_summary.py $(Process) $1
output = output_batch_german/out.$(Process)
error = output_batch_german/error.$(Process)
Log = log.txt
Queue 1
