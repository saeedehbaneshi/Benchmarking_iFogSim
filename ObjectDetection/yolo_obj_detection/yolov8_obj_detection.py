import cv2
import csv
import time
import psutil
import os
import subprocess
from ultralytics import YOLO

# Paths
video_path = "./motion_mask_output.avi"
output_video_path = "Yolov8_output.avi"
csv_output_path = "yolov8_detections.csv"
perf_output_path = "yolov8_perf_output.txt"

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Performance measurement setup
process = psutil.Process()
cpu_start = process.cpu_percent(interval=None)
memory_start = process.memory_info().rss  # Memory usage in bytes
start_time = time.time()

# Run `perf` to measure instructions, cycles, cache misses, etc.
perf_command = "perf stat -e instructions,cycles,cache-misses,branch-misses -- python3 -c 'import time; time.sleep(0.001)'"
with open(perf_output_path, "w") as perf_file:
    subprocess.run(perf_command, shell=True, stderr=perf_file)

# Open CSV file for writing detections
with open(csv_output_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    csv_writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"])

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Perform object detection with verbose=False to suppress terminal output
        results = model(frame, verbose=False)  # Hide console output

        # Iterate over detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class ID
                class_name = model.names[class_id]  # Class name

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write detection to CSV
                csv_writer.writerow([frame_id, x1, y1, x2, y2, confidence, class_id, class_name])

        # Write the frame with detections to the output video
        out.write(frame)

# Measure elapsed time and system performance
elapsed_time = time.time() - start_time
calculated_fps = frame_id / elapsed_time

cpu_end = process.cpu_percent(interval=None)
memory_end = process.memory_info().rss  # Memory usage in bytes

# Extract perf metrics
perf_metrics = {}
with open(perf_output_path, "r") as perf_file:
    for line in perf_file:
        if "instructions" in line:
            perf_metrics["instructions"] = int(line.split()[0].replace(",", ""))
        elif "cycles" in line:
            perf_metrics["cycles"] = int(line.split()[0].replace(",", ""))
        elif "cache-misses" in line:
            perf_metrics["cache_misses"] = int(line.split()[0].replace(",", ""))
        elif "branch-misses" in line:
            perf_metrics["branch_misses"] = int(line.split()[0].replace(",", ""))
        elif "seconds time elapsed" in line:
            perf_metrics["perf_time_elapsed"] = float(line.split()[0])

# Save performance metrics to a CSV file
perf_csv_output_path = "performance_metrics.csv"
with open(perf_csv_output_path, mode="w", newline="") as perf_csv:
    perf_writer = csv.writer(perf_csv)
    perf_writer.writerow(["Metric", "Value"])
    perf_writer.writerow(["Processing Time (s)", elapsed_time])
    perf_writer.writerow(["Frames Processed", frame_id])
    perf_writer.writerow(["FPS", calculated_fps])
    perf_writer.writerow(["CPU Usage (%)", cpu_end - cpu_start])
    perf_writer.writerow(["Memory Usage (MB)", (memory_end - memory_start) / (1024 ** 2)])
    perf_writer.writerow(["Instructions", perf_metrics.get("instructions", "N/A")])
    perf_writer.writerow(["Cycles", perf_metrics.get("cycles", "N/A")])
    perf_writer.writerow(["Cache Misses", perf_metrics.get("cache_misses", "N/A")])
    perf_writer.writerow(["Branch Misses", perf_metrics.get("branch_misses", "N/A")])
    perf_writer.writerow(["Perf Time Elapsed (s)", perf_metrics.get("perf_time_elapsed", "N/A")])

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Object detection completed. Output video saved as: {output_video_path}")
print(f"Detections saved in CSV file: {csv_output_path}")
print(f"Performance metrics saved in CSV file: {perf_csv_output_path}")

