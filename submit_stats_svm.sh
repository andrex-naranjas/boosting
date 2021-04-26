Universe = vanilla
Executable = /usr/bin/python3
Arguments = batch_stats_summary.py $(Process)
output = output_batch/out.$(Process)
error = output_batch/error.$(Process)
Log = log.txt
Queue 100
