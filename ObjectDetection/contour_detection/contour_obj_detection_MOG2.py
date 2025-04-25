#!/usr/bin/env python3
"""
Classical Contour-Based Vehicle Detection on a Mask Video

Input: "mask.avi" (a video file produced by your motion detection microservice)
Output: "output_contour.avi" – video frames with bounding boxes drawn around regions
         that pass our vehicle heuristics.
"""

import cv2
import numpy as np
import time, psutil, os, json

# --- PARAMETERS ---
MIN_AREA = 1500      # minimum contour area to consider (tune based on video resolution)
MIN_ASPECT = 1.0     # minimum width/height ratio (vehicles are generally wider)
MAX_ASPECT = 3.0     # maximum width/height ratio
# You might adjust these thresholds based on your specific environment

# ─── PATHS ─────────────────────────────────────────────────────────────────────
MASK_VIDEO    = "../assets/motion_mask_output.mp4"
OUTPUT_VIDEO  = "../assets/results_contour_method/output_contour_MOG2.avi"
JSON_METRICS  = "../assets/results_contour_method/contour_detection_metrics.json"

# ensure results folder exists
os.makedirs(os.path.dirname(JSON_METRICS), exist_ok=True)

# ─── OPEN INPUT ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(MASK_VIDEO)
if not cap.isOpened():
    print("Error: cannot open mask video")
    exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

input_mb = os.path.getsize(MASK_VIDEO) / (1024**2)

# ─── SET UP OUTPUT ─────────────────────────────────────────────────────────────

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# ─── METRICS COLLECTION ───────────────────────────────────────────────────────
proc       = psutil.Process(os.getpid())
t0_wall    = time.time()
t0_cpu     = sum(proc.cpu_times()[:2])
mem0       = proc.memory_info().rss

frame_count       = 0
detection_frames  = 0
total_detections  = 0

# ─── PROCESS LOOP ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # grayscale/binary cleanup
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find & filter contours
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections_this_frame = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x,y,wc,hc = cv2.boundingRect(c)
        ar = wc/hc if hc>0 else 0
        if ar<MIN_ASPECT or ar>MAX_ASPECT:
            continue
        # draw box & label
        cv2.rectangle(frame, (x,y), (x+wc, y+hc), (0,255,0), 2)
        cv2.putText(frame, "vehicle", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        detections_this_frame += 1

    if detections_this_frame:
        detection_frames += 1
    total_detections += detections_this_frame

    out.write(frame)
    frame_count += 1

# ─── CLEANUP & OUTPUT STATS ───────────────────────────────────────────────────
cap.release()
out.release()

# count output frames
cap2 = cv2.VideoCapture(OUTPUT_VIDEO)
out_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) if cap2.isOpened() else 0
cap2.release()

output_mb = os.path.getsize(OUTPUT_VIDEO) / (1024**2)

t1_wall = time.time()
t1_cpu  = sum(proc.cpu_times()[:2])
mem1    = proc.memory_info().rss

elapsed = t1_wall - t0_wall
cpu_t   = t1_cpu - t0_cpu
cores   = psutil.cpu_count(logical=True) or 1
cpu_pct = (cpu_t/elapsed*100)/cores if elapsed>0 else 0
mem_mb  = (mem1-mem0)/(1024**2)
fps_calc = frame_count/elapsed if elapsed>0 else 0
det_rate = (detection_frames/frame_count*100) if frame_count>0 else 0

# ─── DUMP JSON ────────────────────────────────────────────────────────────────
metrics = {
    "frame_count":        frame_count,
    "total_frames":       total_frames,
    "output_frames":      out_frames,
    "input_mb":           input_mb,
    "output_mb":          output_mb,
    "elapsed_s":          elapsed,
    "fps":                fps_calc,
    "cpu_pct_per_core":   cpu_pct,
    "mem_delta_mb":       mem_mb,
    "detection_frames":   detection_frames,
    "total_detections":   total_detections,
    "detection_rate_pct": det_rate
}

with open(JSON_METRICS, "w") as f:
    json.dump(metrics, f)
