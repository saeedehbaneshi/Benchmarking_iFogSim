# Benchmarking_iFogSim

Scripts and results from benchmarking the modules of a video surveillance application (motion
detection and object detection), plus some general CPU and power benchmarks. This code is not
part of a publication.

## Contents

| Folder | Content |
|---|---|
| `MotionDetection/` | Motion detection on an input video: OpenCV MOG2 (`MOG2_method/`), dense optical flow and rigid flow variants |
| `ObjectDetection/` | Object detection on the motion-detection output: contour-based, YOLOv5n, YOLOv5s and YOLOv8n variants |
| `ObjectTracking/` | Placeholder (notes only) |
| `cpu_benchmarking/` | Results from CoreMark, Phoronix Test Suite, stress-ng and sysbench runs |
| `device_power_benchmarking/` | Idle and sysbench full-load power plots measured with Intel RAPL |

## How the measurements are taken

The MOG2, contour, YOLOv5n and YOLOv5s folders each have a `run_multiple*.sh` script that runs
the Python implementation under
`perf stat` and then calls a `parse_perf*.py` script to collect the counters (for example
instructions and cycles) together with the metrics written by the application.

`MotionDetection/` and `ObjectDetection/` are separate [Poetry](https://python-poetry.org/)
projects. Their dependencies are listed in `pyproject.toml` and pinned in `poetry.lock`.

## Inputs not included

The scripts read their input videos and model weights from `../assets/` (for example
`assets/input_video.mp4`, `assets/motion_mask_output.mp4` and `assets/models/yolov5s.pt`).
These files are not part of this repository.

## Third-party code

- `ObjectDetection/yolo5s_detection_method/yolov5/` is a copy of
  [ultralytics/yolov5](https://github.com/ultralytics/yolov5) added with `git subtree`. It is
  licensed under AGPL-3.0 (see its `LICENSE`).
- `cpu_benchmarking/Phoronix/phoronix-test-suite-master/` is a copy of the
  [Phoronix Test Suite](https://github.com/phoronix-test-suite/phoronix-test-suite), licensed
  under GPL-3.0 (see its `COPYING`).

The rest of this repository does not yet have a license file.

## Contact

Saeedeh Baneshi, baneshisaeedeh@gmail.com
