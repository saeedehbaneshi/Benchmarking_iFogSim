#!/usr/bin/env bash
set -e

RESULT_DIR="../assets/results_yolo5n_method"
mkdir -p "$RESULT_DIR"

for run in {1..10}; do
  echo "=== YOLOv5-n run #$run ==="
  perf stat \
    -e instructions,cycles \
    -o "$RESULT_DIR/perf_yolo5n_${run}.txt" \
    -- python3 yolo5n_detection.py

  python3 parse_perf_yolo5n.py \
    "$RESULT_DIR/yolo5n_metrics.json" \
    "$RESULT_DIR/perf_yolo5n_${run}.txt" \
    "$run"
done

echo "All done: see $RESULT_DIR/perf_metrics.xlsx"

