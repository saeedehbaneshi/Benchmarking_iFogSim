#!/usr/bin/env python3
"""
Classical Contour-Based Vehicle Detection on a Mask Video
with CSV Output for Object Tracking

Input: "mask.avi" (a video file produced by your motion detection microservice)
Output: 
    - "output_contour.avi" – video frames with bounding boxes drawn
    - "detections_contour.csv" – CSV file with detected bounding boxes
"""

import cv2
import numpy as np
import csv

# --- PARAMETERS ---
MIN_AREA = 1500      # minimum contour area to consider (tune based on video resolution)
MIN_ASPECT = 1.0     # minimum width/height ratio (vehicles are generally wider)
MAX_ASPECT = 3.0     # maximum width/height ratio
# You might adjust these thresholds based on your specific environment

# --- INITIALIZE VIDEO CAPTURE and Writer ---
mask_video_path = './motion_mask_output.avi'
output_video_path = './modified_output_contour_MOG2.avi'
csv_output_path = 'detections_contour.csv'  # CSV file to save detections

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

# Open CSV file to store detections (frame_id, x1, y1, x2, y2, score)
csv_file = open(csv_output_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "score"])  # Write header

frame_id = 0  # Frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    frame_id += 1  # Increment frame counter

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
            continue  # Ignore small regions

        # Compute bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Heuristic: assume vehicles have a moderate aspect ratio
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            continue

        # Draw a rectangle on the original mask frame (for visualization)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add label
        cv2.putText(frame, "vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Save detection in CSV (format: frame_id, x1, y1, x2, y2, score)
        csv_writer.writerow([frame_id, x, y, x + w, y + h, 1.0])  # Score is set to 1.0

    # Write the annotated frame to the output video
    out.write(frame)

    # (Optional) Display the frame
    cv2.imshow("Contour-Based Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Process completed: Output video saved as '{output_video_path}', detections saved as '{csv_output_path}'")
