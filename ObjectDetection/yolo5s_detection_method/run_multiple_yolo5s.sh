#!/usr/bin/env bash
set -e

RESULT_DIR="../assets/results_yolo5s_method"
mkdir -p "$RESULT_DIR"

#Running with Torch
#python yolov5/detect.py --weights ../assets/models/yolov5s.pt --source ../assets/motion_mask_output.mp4

for run in {1..1}; do
  echo "=== YOLOv5-s run #$run ==="
  perf stat -r 1 \
    -e instructions,cycles \
    -o "$RESULT_DIR/perf_yolo5s_${run}.txt" \
    -- python yolo5s_detection.py

  python3 parse_perf_yolo5s.py \
    "$RESULT_DIR/perf_yolo5s_${run}.txt" \
    "$run"
done

echo "All done: see $RESULT_DIR/perf_metrics.xlsx"




#python3 parse_perf_yolo5s.py \
#    "$RESULT_DIR/yolo5s_metrics.json" \
#    "$RESULT_DIR/perf_yolo5s_${run}.txt" \
#    "$run"

