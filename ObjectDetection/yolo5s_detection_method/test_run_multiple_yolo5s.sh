#!/usr/bin/env bash
set -e

RESULT_DIR="../assets/results_yolo5s_method"
mkdir -p "$RESULT_DIR"

for run in {1..5}; do
  python3 test_parse_perf_yolo5s.py \
    "$RESULT_DIR/perf_yolo5s_${run}.txt" \
    "$run"
done

echo "All done: see $RESULT_DIR/perf_metrics.xlsx"




#python3 parse_perf_yolo5s.py \
#    "$RESULT_DIR/yolo5s_metrics.json" \
#    "$RESULT_DIR/perf_yolo5s_${run}.txt" \
#    "$run"

