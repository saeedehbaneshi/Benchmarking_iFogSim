#!/usr/bin/env python3
"""
Classical Contour-Based Vehicle Detection on a Mask Video

Input: "mask.avi" (a video file produced by your motion detection microservice)
Output: "output_contour.avi" – video frames with bounding boxes drawn around regions
         that pass our vehicle heuristics.
"""

import cv2
import numpy as np

# --- PARAMETERS ---
MIN_AREA = 1500      # minimum contour area to consider (tune based on video resolution)
MIN_ASPECT = 1.0     # minimum width/height ratio (vehicles are generally wider)
MAX_ASPECT = 3.0     # maximum width/height ratio
# You might adjust these thresholds based on your specific environment

# --- INITIALIZE VIDEO CAPTURE and Writer ---
mask_video_path = './DenseFlow_input_mask/optical_flow_motion_mask.avi'
output_video_path = './DenseFlow_input_mask/output_contour_optical.avi'

cap = cv2.VideoCapture(mask_video_path)
if not cap.isOpened():
    print("Error: Cannot open mask video.")
    exit(1)

# Get video properties for output writer
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object (here we use XVID)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # --- Preprocessing the Mask ---
    # Assume the mask frame is already in grayscale or near-binary.
    # If not, convert to grayscale.
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Threshold the mask (ensure binary image)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Clean the mask: remove noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # --- Contour Extraction ---
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, compute bounding box and apply heuristic filtering
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue  # ignore small regions

        # Compute bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Heuristic: assume vehicles have a moderate aspect ratio
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            continue

        # Optionally, you might add other filters (e.g., size, position, etc.)
        # Draw a rectangle on the original mask frame (for visualization)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # You could also put text "vehicle" next to it
        cv2.putText(frame, "vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # (Optional) Display the frame
    cv2.imshow("Contour-Based Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

