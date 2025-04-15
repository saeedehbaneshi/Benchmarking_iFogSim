import cv2
import numpy as np
import csv
import torch
import torchvision
from bgslibrary import AdaptiveBackgroundLearning  # Ensure this is correctly installed

# Paths
video_path = "../assets/input_video.mp4"
output_filtered_path = "../assets/results_rigid_flow/rigid_motion_filtered.avi"
output_mask_path = "../assets/results_rigid_flow/rigid_motion_mask.avi"
csv_output_path = "../assets/results_rigid_flow/rigid_motion_objects.csv"

# Load Background Subtraction Model
bg_subtractor = AdaptiveBackgroundLearning()

# Load pre-trained Mask R-CNN for Motion Segmentation
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# Video Capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Try "mp4v" or "MJPG" if needed
out_filtered = cv2.VideoWriter(output_filtered_path, fourcc, fps, (frame_width, frame_height))
out_mask = cv2.VideoWriter(output_mask_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Open CSV File for Writing
with open(csv_output_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "score"])  # CSV Header

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if end of video

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = bg_subtractor.apply(gray)  # Motion Segmentation

        # Convert frame for Deep Learning Model
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Apply Motion Detection using Mask R-CNN
        with torch.no_grad():
            detections = model(frame_tensor)

        detected_objects = False  # Track if objects were detected

        for i in range(len(detections[0]["scores"])):
            score = detections[0]["scores"][i].item()
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = detections[0]["boxes"][i].int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Draw white bounding box
                writer.writerow([frame_id, x1, y1, x2, y2, score])
                detected_objects = True  # Mark as detected

        if not detected_objects:
            print(f"Frame {frame_id}: No objects detected.")

        out_filtered.write(frame)  # Save filtered video
        out_mask.write(motion_mask)  # Save motion mask

cap.release()
out_filtered.release()
out_mask.release()
cv2.destroyAllWindows()

print(f"✅ Motion detection done! Outputs:\n- Filtered Video: {output_filtered_path}\n- Motion Mask Video: {output_mask_path}\n- Object Data CSV: {csv_output_path}")

