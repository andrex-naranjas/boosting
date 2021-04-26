Universe = vanilla
Executable = /usr/bin/python3
Arguments = batch_stats_summary.py $(Process)
output = output_batch_german/out.$(Process)
error = output_batch_german/error.$(Process)
Log = log.txt
Queue 100
