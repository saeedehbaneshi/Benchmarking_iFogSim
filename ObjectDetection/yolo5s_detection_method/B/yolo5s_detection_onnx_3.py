#!/usr/bin/env python3
"""
Detect objects using a YOLOv5 ONNX model via ONNX Runtime, no PyTorch dependency.

Supported sources: images, videos, webcam (numeric), directories, glob patterns.
Usage:
    python detect_onnx.py --weights yolov5s.onnx --source input.mp4 --imgsz 640 640 \
        --conf-thres 0.25 --iou-thres 0.45 --device cpu --save-img --save-txt
        

python detect_onnx.py --weights yolov5s.onnx
"""
import argparse
import os
import sys
import glob
import time
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Supported file extensions
IMG_EXTS = ['jpg','jpeg','png','bmp','tiff']
VID_EXTS = ['mp4','avi','mov','mkv','flv']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--source', type=str, default='0', help='file/dir/glob/webcam')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[640,640], help='Inference size H W')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='Only CPU is supported')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--save-img', action='store_true', help='Save annotated images/videos')
    parser.add_argument('--save-txt', action='store_true', help='Save detection .txt labels')
    parser.add_argument('--save-csv', action='store_true', help='Save detection .csv')
    parser.add_argument('--save-conf', action='store_true', help='Include confidence in labels')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped detections')
    parser.add_argument('--nosave', action='store_true', help='Do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class indices')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing project/name')
    parser.add_argument('--line-thickness', type=int, default=3, help='Bounding box thickness')
    parser.add_argument('--hide-labels', action='store_true', help='Hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='Hide confidences')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz = (opt.imgsz[0], opt.imgsz[1])
    return opt


def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    nh, nw = int(h0*r), int(w0*r)
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    dw, dh = (new_shape[1]-nw)//2, (new_shape[0]-nh)//2
    pad[dh:dh+nh, dw:dw+nw] = im_resized
    return pad, r, dw, dh


def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    # predictions: [N,6] (x1,y1,x2,y2,conf,cls)
    if not len(predictions):
        return []
    # Filter by conf
    mask = predictions[:,4] >= conf_thres
    pred = predictions[mask]
    if not len(pred):
        return []
    # If class filtering
    if classes:
        mask = np.isin(pred[:,5].astype(int), classes)
        pred = pred[mask]
    # Perform NMS
    boxes = pred[:,:4]
    scores = pred[:,4]
    classes_inds = pred[:,5] if not agnostic else np.zeros_like(pred[:,5])
    # Sort by score
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        rem_areas = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) +
                    (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1]) - inter
        iou = inter / rem_areas
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds+1]
    return pred[keep]


def get_source_list(source):
    # Returns list of paths or ['0'] for webcam
    if source.isnumeric():
        return [int(source)]
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMG_EXTS+VID_EXTS:
            files += glob.glob(str(p/'*.'+ext))
        return sorted(files)
    if '*' in source:
        return sorted(glob.glob(source))
    if p.is_file():
        return [source]
    raise ValueError(f'Invalid source {source}')


def main(opt):
    # Setup
    save_dir = Path(opt.project)/opt.name
    save_dir.mkdir(parents=True, exist_ok=opt.exist_ok)
    is_webcam = False
    sources = get_source_list(opt.source)
    for s in sources:
        if isinstance(s, int): is_webcam = True
    # Load ONNX model
    sess = ort.InferenceSession(opt.weights, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    # Loop
    for source in sources:
        cap = cv2.VideoCapture(source) if not isinstance(source,int) else cv2.VideoCapture(source)
        out_path = str(save_dir/f'{Path(str(source)).stem}.mp4') if opt.save_img and not is_webcam else None
        writer = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            # stride
            # preprocess
            img, r, dw, dh = letterbox(frame, new_shape=opt.imgsz)
            img = img[:,:,::-1].astype(np.float32)/255.0
            img = np.transpose(img,(2,0,1))[None,...]
            # inference
            pred = sess.run(None, {inp_name: img})[0]
            # if model includes NMS, pred-> [1,N,6]
            if pred.ndim==3:
                det = pred[0]
            else:
                det = pred
            # NMS if needed
            det = non_max_suppression(det, opt.conf_thres, opt.iou_thres,
                                      classes=opt.classes, agnostic=opt.agnostic_nms)
            # Process
            for *xyxy, conf, cls in det.tolist():
                if opt.hide_labels:
                    label = ''
                else:
                    label = f'{int(cls)}'
                if not opt.hide_conf:
                    label += f' {conf:.2f}'
                x1,y1,x2,y2 = map(int,xyxy)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),opt.line_thickness)
                if label:
                    cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            # save view
            if opt.view_img:
                cv2.imshow(str(source), frame); cv2.waitKey(1)
            if opt.save_img:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    h,w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_path,fourcc,fps,(w,h))
                writer.write(frame)
        cap.release()
        if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

