Universe = vanilla
Executable = /usr/bin/python3
Arguments = batch_stats_summary.py $(Process) $("titanic")
output = output_batch_test/out.$(Process)
error = output_batch_test/error.$(Process)
Log = log.txt
Queue 1
