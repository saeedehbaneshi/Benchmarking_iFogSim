#!/usr/bin/env python3
"""
Vehicle detection on a grayscale traffic video via ONNX Runtime,
with JSON-only metrics.  No .pt or ultralytics needed.
"""

import os
import time
import json
import cv2
import psutil
import numpy as np
import onnxruntime as ort

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_VIDEO   = "../assets/motion_mask_output.mp4"
OUTPUT_VIDEO  = "../assets/results_yolo5s_method/output_yolo5s_onnx.mp4"
JSON_METRICS  = "../assets/results_yolo5s_method/yolo5s_metrics_onnx.json"
ONNX_MODEL    = "../assets/models/yolov5s.onnx"
IMAGE_SIZE    = 640            # model’s expected H/W
CONF_THRESH   = 0.10
IOU_THRESH    = 0.45

# COCO class names (80 classes)
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

# Vehicle classes we care about
VEHICLES = {"car","bus","truck","motorcycle","bicycle","train"}

os.makedirs(os.path.dirname(JSON_METRICS), exist_ok=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def letterbox(im, new_shape=(IMAGE_SIZE, IMAGE_SIZE), color=(114,114,114)):
    """Resize-with-padding, return padded image, scale, pad (w,h)."""
    h0, w0 = im.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    nh, nw = int(h0*r), int(w0*r)
    im_resized = cv2.resize(im, (nw,nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    dw, dh = (new_shape[1]-nw)//2, (new_shape[0]-nh)//2
    pad[dh:dh+nh, dw:dw+nw] = im_resized
    return pad, r, dw, dh

def nms_numpy(boxes, scores, iou_thresh=IOU_THRESH):
    """
    Pure-Numpy NMS. boxes = [N,4], scores=[N].
    Returns list of kept indices.
    """
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds+1]
    return keep

# ─── LOAD ONNX MODEL ───────────────────────────────────────────────────────────

sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# ─── OPEN VIDEO & METRICS SETUP ─────────────────────────────────────────────────

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS     = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FR= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
INPUT_MB= os.path.getsize(INPUT_VIDEO)/(1024**2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W,H))

proc    = psutil.Process(os.getpid())
t0_wall = time.time()
t0_cpu  = sum(proc.cpu_times()[:2])
mem0    = proc.memory_info().rss

frame_count      = 0
detection_frames = 0
total_detections = 0

# ─── PROCESS FRAMES ────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ensure 3-channel BGR
    if frame.ndim==2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # preprocess
    img_pad, scale, pad_w, pad_h = letterbox(frame)
    img = img_pad[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, norm
    img = np.transpose(img, (2,0,1))[None,:,:,:]        # HWC→NCHW

    # inference
    outs = sess.run(None, {input_name: img})
    dets = outs[0][0]  # [x1,y1,x2,y2,conf,cls]

    # filter low confidence
    mask = dets[:,4] > CONF_THRESH
    dets = dets[mask]

    # unpad & unscale
    if dets.size:
        dets[:,[0,2]] = (dets[:,[0,2]] - pad_w) / scale
        dets[:,[1,3]] = (dets[:,[1,3]] - pad_h) / scale

    # class-wise NMS
    final = []
    for cls in np.unique(dets[:,5].astype(int)):
        cls_mask = dets[:,5]==cls
        boxes  = dets[cls_mask,:4]
        scores = dets[cls_mask,4]
        keep   = nms_numpy(boxes, scores)
        final.append(dets[cls_mask][keep])
    if final:
        dets = np.vstack(final)
    else:
        dets = np.zeros((0,6))

    # draw & count vehicles
    det_count = 0
    for *box, conf, cls in dets.tolist():
        name = COCO_NAMES[int(cls)]
        if name not in VEHICLES:
            continue
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{name} {conf:.2f}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        det_count += 1

    if det_count:
        detection_frames += 1
    total_detections += det_count

    out.write(frame)
    frame_count += 1

# ─── FINALIZE & METRICS ────────────────────────────────────────────────────────

cap.release()
out.release()
cap2 = cv2.VideoCapture(OUTPUT_VIDEO)
out_fc = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) if cap2.isOpened() else 0
cap2.release()
OUTPUT_MB = os.path.getsize(OUTPUT_VIDEO)/(1024**2)

t1_wall = time.time()
t1_cpu  = sum(proc.cpu_times()[:2])
mem1    = proc.memory_info().rss

elapsed_s = t1_wall - t0_wall
cpu_time  = t1_cpu - t0_cpu
cores     = psutil.cpu_count(logical=True) or 1
cpu_pct   = (cpu_time/elapsed_s*100)/cores if elapsed_s>0 else 0
mem_delta = (mem1 - mem0)/(1024**2)
fps_calc  = frame_count/elapsed_s if elapsed_s>0 else 0
det_rate  = (detection_frames/frame_count*100) if frame_count>0 else 0

metrics = {
    "frame_count":       frame_count,
    "total_frames":      TOTAL_FR,
    "output_frames":     out_fc,
    "input_mb":          INPUT_MB,
    "output_mb":         OUTPUT_MB,
    "elapsed_s":         elapsed_s,
    "fps":               fps_calc,
    "cpu_pct_per_core":  cpu_pct,
    "mem_delta_mb":      mem_delta,
    "detection_frames":  detection_frames,
    "total_detections":  total_detections,
    "detection_rate_pct":det_rate,
}

with open(JSON_METRICS,"w") as f:
    json.dump(metrics, f, indent=2)

print("Done. Metrics saved to", JSON_METRICS)

