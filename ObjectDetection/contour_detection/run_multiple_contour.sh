#!/usr/bin/env bash
set -e

RESULT_DIR="../assets/results_contour_method"
mkdir -p "$RESULT_DIR"

for run in {1..10}; do
  echo "===== Contour detection run #$run ====="

  perf stat \
    -e instructions,cycles \
    -o "$RESULT_DIR/perf_contour_${run}.txt" \
    -- python3 contour_obj_detection_MOG2.py

  python3 parse_perf_contour.py \
    "$RESULT_DIR/contour_detection_metrics.json" \
    "$RESULT_DIR/perf_contour_${run}.txt" \
    "$run"

done

echo "All runs done. Results in $RESULT_DIR/perf_metrics.xlsx"

