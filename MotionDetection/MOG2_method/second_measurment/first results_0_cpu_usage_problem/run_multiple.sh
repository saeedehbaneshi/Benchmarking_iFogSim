#!/bin/bash

# Number of runs
NUM_RUNS=5

# Path to your Python script
PYTHON_SCRIPT="second_motion_detection_MOG2.py"

# Path to the perf output file
PERF_OUTPUT="second_perf_output.txt"

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Running iteration $i of $NUM_RUNS..."
    perf stat -o $PERF_OUTPUT python3 $PYTHON_SCRIPT
    python3 parse_perf.py motion_detection_metrics.json $PERF_OUTPUT $i
    echo "Completed iteration $i"
    sleep 1
done

echo "All runs completed. Check perf_metrics.xlsx for results."
