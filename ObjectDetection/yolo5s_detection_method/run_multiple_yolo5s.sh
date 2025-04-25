#!/usr/bin/env bash
set -e

RESULT_DIR="../assets/results_yolo5s_method"
mkdir -p "$RESULT_DIR"

for run in {1..10}; do
  echo "=== YOLOv5-s run #$run ==="
  perf stat \
    -e instructions,cycles \
    -o "$RESULT_DIR/perf_yolo5s_${run}.txt" \
    -- python3 yolo5s_detection.py

  python3 parse_perf_yolo5s.py \
    "$RESULT_DIR/yolo5s_metrics.json" \
    "$RESULT_DIR/perf_yolo5s_${run}.txt" \
    "$run"
done

echo "All done: see $RESULT_DIR/perf_metrics.xlsx"

