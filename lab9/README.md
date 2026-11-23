# Computer Vision Lab 09: Modern Object Detection & Real-Time Pose Estimation

**Institution:** National University of Computer & Emerging Sciences (FAST-NUCES), Karachi  
**Department:** Department of Artificial Intelligence  
**Course:** Computer Vision (AI-4002)  
**Student Name:** Muhammad John Raza  
**Roll Number:** 23k-0069  
**Course Instructor:** Talha Shahid  

---

## Table of Contents
1. [Lab Objectives & Task Overview](#lab-objectives--task-overview)
2. [Theoretical Foundations & Formulations](#theoretical-foundations--formulations)
   - [YOLO (You Only Look Once)](#yolo-you-only-look-once)
   - [SSD (Single Shot MultiBox Detector)](#ssd-single-shot-multibox-detector)
   - [RetinaNet & Focal Loss](#retinanet--focal-loss)
   - [Human Pose Estimation & Joint Angles](#human-pose-estimation--joint-angles)
3. [Implemented Tasks Breakdown](#implemented-tasks-breakdown)
   - [Task 1: Live Webcam Lab Asset Detection (YOLO)](#task-1-live-webcam-lab-asset-detection-yolo)
   - [Task 2: Comparative Detection (YOLOv3 vs. SSD vs. RetinaNet)](#task-2-comparative-detection-yolov3-vs-ssd-vs-retinanet)
   - [Task 3: Speed (FPS), Confidence, Model Size & Latency Benchmarks](#task-3-speed-fps-confidence-model-size--latency-benchmarks)
   - [Task 4: Analytical Discussion & Performance Evaluation](#task-4-analytical-discussion--performance-evaluation)
   - [Task 5: Real-Time Pose Estimation from Webcam Feed](#task-5-real-time-pose-estimation-from-webcam-feed)
4. [Directory & File Structure](#directory--file-structure)
5. [How to Run](#how-to-run)
6. [Experimental Results & Verification](#experimental-results--verification)

---

## Lab Objectives & Task Overview

Per **Section 7: Lab Task** of the Lab 09 manual:
1. **Live Webcam Lab Asset Detection:** Utilize YOLO to detect and classify laboratory equipment (laptops, monitors, keyboards, mice, chairs, books, bottles, cups, cell phones, students).
2. **Multi-Model Object Detection:** Implement and evaluate **YOLOv3**, **SSD**, and **RetinaNet** across a common benchmark dataset with bounding boxes, labels, and confidence scores.
3. **Empirical Benchmarks:** Measure and log Detection Speed (FPS), Mean Confidence across object categories, Model Footprint (MB / Parameters), and Per-Frame Latency.
4. **Architectural Discussion:** Rigorously analyze model suitability for:
   - Real-Time Detection
   - Small Object Recognition
   - Resource-Limited (CPU-Only) Deployments
5. **Real-Time Human Pose Estimation:** Capture video, detect human skeletons frame-by-frame, compute dynamic joint angles (e.g., Left and Right Elbow angles), and output a real-time coordinates stream.

---

## Theoretical Foundations & Formulations

### YOLO (You Only Look Once)
YOLO frames object detection as a single regression problem, dividing the input image into an $S \times S$ grid. For each grid cell, the network predicts $B$ bounding boxes with parameterized coordinates and class probabilities:
$$b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y$$
$$b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}$$
$$\text{Pr}(\text{Class}_i \mid \text{Object}) \times \text{IoU}(\text{truth}, \text{pred}) = \text{Confidence Score}$$

### SSD (Single Shot MultiBox Detector)
SSD eliminates proposal generation entirely by predicting default anchor bounding boxes over multiple feature map resolutions (e.g. $19 \times 19, 10 \times 10, 5 \times 5, 3 \times 3, 1 \times 1$). MultiBox objective minimizes a combined loss:
$$\mathcal{L}(x, c, l, g) = \frac{1}{N} \left( \mathcal{L}_{\text{conf}}(x, c) + \alpha \mathcal{L}_{\text{loc}}(x, l, g) \right)$$
Using **Depthwise Separable Convolutions** in MobileNetV3 backbones, SSD achieves ultra-low computational overhead suitable for mobile and edge CPUs.

### RetinaNet & Focal Loss
Standard single-stage detectors suffer from extreme foreground-background class imbalance (~100,000 candidate locations with >99% easy negatives). RetinaNet resolves this with **Focal Loss**:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where:
- $p_t$ is the model's estimated probability for the ground-truth class.
- $\gamma$ is the focusing parameter ($\gamma = 2.0$).
- When an easy negative has $p_t \to 0.99$, the modulating factor $(1 - p_t)^2 = 0.0001$ suppresses its loss by a factor of 10,000, focusing gradients exclusively on hard, tiny object proposals extracted via the **Feature Pyramid Network (FPN)**.

### Human Pose Estimation & Joint Angles
Human pose estimation tracks 17 COCO keypoints (joints). For any three sequential joints $A$, $B$ (pivot joint), and $C$, the joint angle $\theta$ is formulated via vector trigonometry:
$$\vec{u} = \vec{A} - \vec{B}, \quad \vec{v} = \vec{C} - \vec{B}$$
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
$$\theta = \arccos\left(\text{clamp}(\cos(\theta), -1.0, 1.0)\right) \times \frac{180}{\pi}$$
Specifically, the **Elbow Angle** is calculated using Shoulder ($A$), Elbow ($B$), and Wrist ($C$).

---

## Directory & File Structure

```
lab9/
├── assets/                                  # Staged test images and simulated fallback video streams
│   ├── lab_assets_01.jpg                    # Desk setup: laptop, keyboard, mouse, bottle, book
│   ├── lab_assets_02.jpg                    # Workstation with student working
│   ├── lab_assets_03.jpg                    # Multi-object desk peripherals
│   ├── pose_person.jpg                      # Full-body pose image
│   ├── lab_webcam_stream.mp4                # 60-frame simulated lab webcam feed
│   └── pose_webcam_stream.mp4               # 60-frame simulated human pose feed
├── models/                                  # Lightweight pre-trained detection & pose models
│   ├── yolov3-tinyu.pt                      # YOLOv3-tiny weights (23.3 MB)
│   ├── yolov8n.pt                           # YOLOv8 nano weights (6.2 MB)
│   └── yolov8n-pose.pt                      # YOLOv8 pose estimation weights (6.5 MB)
├── outputs/                                 # Visualizations, charts, videos, and logs
│   ├── comparison_yolov3_ssd_retinanet.png  # Grand 3x3 multi-model detection comparison grid
│   ├── benchmark_metrics.csv                # Speed, FPS, Latency, Size, Params summary table
│   ├── benchmark_class_confidences.csv      # Per-class confidence breakdown
│   ├── benchmark_comparison_charts.png      # 4-panel publication benchmark graphs
│   ├── performance_radar_chart.png          # 6-axis model capability radar comparison
│   ├── performance_discussion_report.txt    # Technical evaluation & scenario selection report
│   ├── yolo_webcam_lab_assets.mp4           # Task 1 live detection video with HUD
│   ├── yolo_webcam_detection_snapshot.png   # Task 1 detection snapshot
│   ├── realtime_pose_skeleton_feed.mp4      # Task 5 annotated skeleton video feed
│   ├── pose_estimation_skeleton_snapshot.png# Task 5 skeleton snapshot with joint angles
│   └── pose_coordinates_stream.csv          # Real-time joint coordinates & elbow angles
├── generate_assets.py                       # Downloads and synthesizes real lab test assets
├── task1_yolo_webcam_detection.py           # Task 1 live webcam lab asset detection
├── task2_multi_model_detection.py           # Task 2 comparative detection (YOLOv3 vs SSD vs RetinaNet)
├── task3_benchmark_metrics.py               # Task 3 benchmarking speed, FPS, size, latency
├── task4_performance_analysis.py            # Task 4 theoretical discussion & radar plot
├── task5_realtime_pose_estimation.py        # Task 5 live pose estimation & joint angles
├── run_all.py                               # Master end-to-end execution pipeline
├── Lab09_Solutions.ipynb                    # Executed Jupyter notebook with all outputs
└── README.md                                # Comprehensive documentation
```

---

## How to Run

### Option 1: Execute Entire Pipeline End-to-End
```bash
python lab9/run_all.py
```

### Option 2: Run Individual Tasks
```bash
# Task 1: Live Webcam Lab Asset Detection
python lab9/task1_yolo_webcam_detection.py

# Task 2: Comparative Detection (YOLOv3 vs. SSD vs. RetinaNet)
python lab9/task2_multi_model_detection.py

# Task 3: Measure & Record Benchmarks (FPS, Latency, Model Size)
python lab9/task3_benchmark_metrics.py

# Task 4: Performance Discussion & Radar Chart
python lab9/task4_performance_analysis.py

# Task 5: Real-Time Pose Estimation & Elbow Angle Tracking
python lab9/task5_realtime_pose_estimation.py
```

---

## Experimental Results & Verification

### Benchmark Summary Table
| Model | Backbone | Parameters | Disk Size | Latency (CPU) | Processing Speed | Best Suited For |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **SSD-MobileNetV3** | MobileNetV3-Large | **3.4 M** | **13.4 MB** | **~210 ms** | **4.8 FPS (CPU) / >60 FPS (GPU)** | **Real-Time & CPU / Edge IoT** |
| **YOLOv3-tiny** | Darknet / Tiny | 8.8 M | 23.3 MB | ~360 ms | 2.8 FPS (CPU) / >45 FPS (GPU)** | **Balanced Edge Video Analytics** |
| **RetinaNet** | ResNet-50-FPN | 34.0 M | 130.0 MB | ~7,400 ms | 0.14 FPS (CPU) / >25 FPS (GPU)** | **Small & Dense Object Detection** |

### Task 4 Summary Findings:
1. **Real-Time Detection:** **SSD / SSDLite-MobileNetV3** wins due to single-pass anchor regression and low latency.
2. **Small Object Recognition:** **RetinaNet** wins due to the combination of multi-scale Feature Pyramid Networks (FPN) and Focal Loss downweighting background negatives.
3. **Resource-Limited Systems:** **SSD-MobileNetV3** wins with only 3.4M parameters and depthwise separable convolutions minimizing RAM and compute.
