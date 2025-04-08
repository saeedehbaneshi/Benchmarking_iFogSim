import cv2
import time
import psutil
import os
import json

# Adjust these paths as needed
VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "motion_mask_output.mp4"

def main():
    print("=== DEBUG: Starting motion detection script ===")
    
    # 1. Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Unable to open video: {VIDEO_PATH}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    input_file_size = 0.0
    if os.path.exists(VIDEO_PATH):
        input_file_size = os.path.getsize(VIDEO_PATH) / (1024 ** 2)
    
    # 2. Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps,
                          (frame_width, frame_height), False)
    if not out.isOpened():
        print(f"Error: Unable to create output: {OUTPUT_VIDEO_PATH}")
        cap.release()
        return
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    # 3. Start measuring time & CPU
    start_time = time.time()
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # Reset the clock
    memory_start = process.memory_info().rss
    
    frame_count = 0
    motion_frames = 0
    total_motion_area = 0
    
    # 4. Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = bg_subtractor.apply(gray)
        out.write(motion_mask)
        
        motion_area = (motion_mask == 255).sum()
        if motion_area > 0:
            motion_frames += 1
            total_motion_area += motion_area
        
        frame_count += 1
    
    elapsed_time = time.time() - start_time
    calculated_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    cpu_usage_percentage = process.cpu_percent(interval=elapsed_time)
    number_of_cores = psutil.cpu_count()
    cpu_usage_per_core = cpu_usage_percentage / number_of_cores
    memory_end = process.memory_info().rss
    
    cap.release()
    out.release()
    
    # 5. Output video info
    output_file_size = 0.0
    output_frame_count = 0
    if os.path.exists(OUTPUT_VIDEO_PATH):
        output_file_size = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 ** 2)
        out_cap = cv2.VideoCapture(OUTPUT_VIDEO_PATH)
        output_frame_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_cap.release()
    
    motion_percentage = (motion_frames / frame_count) * 100 if frame_count > 0 else 0
    mem_usage_mb = (memory_end - memory_start) / (1024 ** 2)
    
    # Save motion detection metrics to JSON file
    motion_detection_metrics = {
        "frame_count": int(frame_count),
        "elapsed_time": elapsed_time,
        "calculated_fps": calculated_fps,
        "input_file_size": input_file_size,
        "output_file_size": output_file_size,
        "cpu_usage_per_core": cpu_usage_per_core,
        "mem_usage_mb": mem_usage_mb,
        "motion_frames": int(motion_frames),
        "total_motion_area": int(total_motion_area),
        "motion_percentage": motion_percentage,
        "frame_count_total": int(frame_count_total),
        "output_frame_count": int(output_frame_count)
    }
    with open("motion_detection_metrics.json", "w") as f:
        json.dump(motion_detection_metrics, f)
    
    # Print summary (optional, for debugging)
    print("=== Summary ===")
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds (FPS: {calculated_fps:.2f})")
    print(f"Motion frames: {motion_frames}, motion %: {motion_percentage:.2f}")
    print(f"Input size (MB): {input_file_size:.2f} | Output size (MB): {output_file_size:.2f}")
    print(f"CPU usage per core: {cpu_usage_per_core:.2f}%")
    print(f"Memory usage (MB): {mem_usage_mb:.2f}")

if __name__ == "__main__":
    main()
