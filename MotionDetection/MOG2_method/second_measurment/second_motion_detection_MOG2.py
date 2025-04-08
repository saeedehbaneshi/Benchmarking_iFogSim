import cv2
import time
import psutil
import os
import json

# Adjust these paths as needed
VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "motion_mask_output.mp4"

def main():
    # 1. Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return  # no printing, to keep overhead minimal

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_file_size = 0.0
    if os.path.exists(VIDEO_PATH):
        input_file_size = os.path.getsize(VIDEO_PATH) / (1024 ** 2)

    # 2. Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps,
        (frame_width, frame_height), isColor=False
    )
    if not out.isOpened():
        cap.release()
        return

    # Create MOG2 Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    # 3. Start measuring time & CPU
    process = psutil.Process(os.getpid())
    start_wall_time = time.time()

    # Record CPU times (user + system) at the start
    start_cpu_times = process.cpu_times()
    start_user = start_cpu_times.user
    start_system = start_cpu_times.system

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

    end_wall_time = time.time()
    elapsed_time = end_wall_time - start_wall_time

    # Record CPU times at the end
    end_cpu_times = process.cpu_times()
    end_user = end_cpu_times.user
    end_system = end_cpu_times.system

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

    motion_percentage = (motion_frames / frame_count * 100) if frame_count > 0 else 0
    mem_usage_mb = 0.0
    memory_end = process.memory_info().rss
    mem_usage_mb = (memory_end - memory_start) / (1024 ** 2)

    # 6. CPU usage calculations via total CPU time
    total_cpu_time = (end_user - start_user) + (end_system - start_system)
    # average CPU usage across all cores:
    usage_across_cores = (total_cpu_time / elapsed_time) * 100 if elapsed_time > 0 else 0
    # per-core usage:
    num_cores = psutil.cpu_count()
    cpu_usage_per_core = usage_across_cores / num_cores if num_cores > 0 else 0

    # 7. Save motion detection metrics to JSON
    motion_detection_metrics = {
        "frame_count": int(frame_count),
        "elapsed_time": elapsed_time,
        "calculated_fps": (frame_count / elapsed_time) if elapsed_time > 0 else 0,
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

    # Write JSON
    with open("motion_detection_metrics.json", "w") as f:
        json.dump(motion_detection_metrics, f)

    # 8. (Optional) if you want no console prints, remove or comment out:
    # print(f"Frames processed: {frame_count}, CPU usage per core: {cpu_usage_per_core}")

if __name__ == "__main__":
    main()

