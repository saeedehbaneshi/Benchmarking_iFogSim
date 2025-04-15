#!/bin/bash

# Number of runs
NUM_RUNS=5

# Path to your Python script
PYTHON_SCRIPT="motion_detection_MOG2.py"

# Path to the perf output file
PERF_OUTPUT="../assets/results_MOG2_method/perf_output.txt"

# Path to the json output file
JSON_OUTPUT="../assets/results_MOG2_method/motion_detection_metrics.json"

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Running iteration $i of $NUM_RUNS..."
    perf stat -o $PERF_OUTPUT python3 $PYTHON_SCRIPT
    python3 parse_perf.py $JSON_OUTPUT $PERF_OUTPUT $i
    echo "Completed iteration $i"
    sleep 1
done

echo "All runs completed. Check perf_metrics.xlsx for results."
