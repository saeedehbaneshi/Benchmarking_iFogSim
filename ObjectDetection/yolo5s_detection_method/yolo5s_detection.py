#!/usr/bin/env python3
"""
YOLOv5-n Vehicle Detection on a grayscale traffic video
with JSON-only metrics (no parsing overhead).
"""

import logging
import cv2
import time
import psutil
import os
import json
from ultralytics import YOLO

# ─── SILENCE ULTRALYTICS LOGGER ───────────────────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── PATHS & CONFIG ────────────────────────────────────────────────────────────
INPUT_VIDEO   = "../assets/motion_mask_output.mp4"
OUTPUT_VIDEO  = "../assets/results_yolo5s_method/output_yolo5s.mp4"
JSON_METRICS  = "../assets/results_yolo5s_method/yolo5s_metrics.json"
WEIGHTS       = "../assets/models/yolov5s.pt"

# COCO vehicle classes
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}

# ensure output dir exists
os.makedirs(os.path.dirname(JSON_METRICS), exist_ok=True)

# load model quietly
model = YOLO(WEIGHTS, verbose=False)

# open input video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS      = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FR = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
INPUT_MB = os.path.getsize(INPUT_VIDEO) / (1024**2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

# ─── START METRICS ─────────────────────────────────────────────────────────────
proc     = psutil.Process(os.getpid())
t0_wall  = time.time()
t0_cpu   = sum(proc.cpu_times()[:2])
mem0     = proc.memory_info().rss

frame_count      = 0
detection_frames = 0
total_detections = 0

# ─── DETECTION LOOP ────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ensure 3-channel for YOLO
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame

    # inference quietly
    res = model.predict(img, conf=0.10, iou=0.45, verbose=False)[0]

    dets = 0
    for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        name = model.names[int(cls_id)]
        if name not in VEHICLES:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        dets += 1

    if dets:
        detection_frames += 1
    total_detections += dets

    out.write(img)
    frame_count += 1

# ─── CLEANUP ───────────────────────────────────────────────────────────────────
cap.release()
out.release()

# count output frames
cap2   = cv2.VideoCapture(OUTPUT_VIDEO)
out_fc = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) if cap2.isOpened() else 0
cap2.release()
OUTPUT_MB = os.path.getsize(OUTPUT_VIDEO) / (1024**2)

# ─── END METRICS ───────────────────────────────────────────────────────────────
t1_wall = time.time()
t1_cpu  = sum(proc.cpu_times()[:2])
mem1    = proc.memory_info().rss

elapsed_s  = t1_wall - t0_wall
cpu_time   = t1_cpu - t0_cpu
cores      = psutil.cpu_count(logical=True) or 1
cpu_pct    = (cpu_time / elapsed_s * 100) / cores if elapsed_s > 0 else 0
mem_delta  = (mem1 - mem0) / (1024**2)
fps_calc   = frame_count / elapsed_s if elapsed_s > 0 else 0
det_rate   = (detection_frames / frame_count * 100) if frame_count > 0 else 0

metrics = {
    "frame_count":        frame_count,
    "total_frames":       TOTAL_FR,
    "output_frames":      out_fc,
    "input_mb":           INPUT_MB,
    "output_mb":          OUTPUT_MB,
    "elapsed_s":          elapsed_s,
    "fps":                fps_calc,
    "cpu_pct_per_core":   cpu_pct,
    "mem_delta_mb":       mem_delta,
    "detection_frames":   detection_frames,
    "total_detections":   total_detections,
    "detection_rate_pct": det_rate,
}

with open(JSON_METRICS, "w") as f:
    json.dump(metrics, f)

