import cv2
import numpy as np
import csv
from bgslibrary import AdaptiveBackgroundLearning

# Paths
video_path = "./Sample_video.mp4"
output_filtered_path = "./rigid_motion_filtered.avi"
output_mask_path = "./rigid_motion_mask.avi"
csv_output_path = "./rigid_motion_objects.csv"

# Load Background Subtraction Model
bg_subtractor = AdaptiveBackgroundLearning()

# Video Capture
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Try "MP4V" if needed
out_filtered = cv2.VideoWriter(output_filtered_path, fourcc, fps, (frame_width, frame_height))
out_mask = cv2.VideoWriter(output_mask_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Open CSV File for Writing
with open(csv_output_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "x1", "y1", "x2", "y2"])  # Only motion bounding boxes

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = bg_subtractor.apply(gray)  # Apply motion detection

        # Find contours of moving objects
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small noise
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Draw bounding box
                writer.writerow([frame_id, x, y, x + w, y + h])  # Save to CSV
        
        if motion_detected:
            print(f"Frame {frame_id}: Motion detected")

        # Write frames to output videos
        out_filtered.write(frame)
        out_mask.write(motion_mask)

cap.release()
out_filtered.release()
out_mask.release()
print(f"Motion detection completed. Outputs saved: {output_filtered_path}, {output_mask_path}, {csv_output_path}")

