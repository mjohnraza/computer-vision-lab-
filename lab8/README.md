# Computer Vision Lab 08: Object Detection using R-CNN, Fast R-CNN & Faster R-CNN

**Institution:** National University of Computer & Emerging Sciences (FAST-NUCES), Karachi  
**Department:** Department of Artificial Intelligence  
**Course:** Computer Vision (AI-4002)  
**Student Name:** Muhammad John Raza  
**Roll Number:** 23k-0069  
**Course Instructor:** Talha Shahid  

---

## Table of Contents
1. [Lab Objectives & Task Requirements](#lab-objectives--task-requirements)
2. [Dataset: Autonomous Traffic Object Detection](#dataset-autonomous-traffic-object-detection)
3. [R-CNN Architecture & Pipeline](#r-cnn-architecture--pipeline)
   - [Step 1: Region Proposal Generation (Selective Search)](#step-1-region-proposal-generation-selective-search)
   - [Step 2: IoU Matching & Proposal Labeling](#step-2-iou-matching--proposal-labeling)
   - [Step 3: Region Warping & Deep CNN Feature Extraction](#step-3-region-warping--deep-cnn-feature-extraction)
   - [Step 4: Multi-Task Training (Classification + BBox Regression)](#step-4-multi-task-training-classification--bbox-regression)
   - [Step 5: Inference & Non-Maximum Suppression (NMS)](#step-5-inference--non-maximum-suppression-nms)
4. [Architectural Comparison: CNN vs. R-CNN vs. Fast R-CNN vs. Faster R-CNN](#architectural-comparison)
5. [Directory & File Structure](#directory--file-structure)
6. [How to Run](#how-to-run)
7. [Experimental Results & Visualizations](#experimental-results--visualizations)

---

## Lab Objectives & Task Requirements

### Task 6.1 (From Lab Manual):
> **6.1 Object Detection Using Regional Convolutional Neural Networks (R-CNN)**  
> *Note: Do not use the mentioned BCCD dataset; use another dataset of your choice.*  
> **Objective:** Implement object detection using a Regional Convolutional Neural Network (R-CNN) on a dataset of your choice. R-CNN is a classic object detection method that combines region proposal, feature extraction, and object classification to identify and localize objects within images.

---

## Dataset: Autonomous Traffic Object Detection
Per the task requirement (*"Do not use BCCD; use another dataset of your choice"*), we implemented an **Autonomous Driving & Traffic Scene Object Detection** dataset.

- **Classes:** `Car`, `Pedestrian`, `Traffic_Sign`, and `Background`.
- **Annotations Schema (CSV format, matching Section 4 of Lab Manual):**
  - `image_names`: Image file name (e.g. `1.jpg`)
  - `cell_type`: Object class name
  - `xmin`: Left horizontal pixel coordinate
  - `xmax`: Right horizontal pixel coordinate
  - `ymin`: Top vertical pixel coordinate
  - `ymax`: Bottom vertical pixel coordinate

---

## R-CNN Architecture & Pipeline

### Step 1: Region Proposal Generation (Selective Search)
Selective Search replaces exhaustive sliding windows with hierarchical grouping:
- Computes over-segmented superpixels using Felzenszwalb-Huttenlocher graph-based segmentation.
- Greedily merges similar regions based on **Color Similarity**, **Texture Similarity**, **Size Compatibility**, and **Shape/Fill Compatibility**.
- Proposes candidate bounding boxes $(x, y, w, h)$.

### Step 2: IoU Matching & Proposal Labeling
Intersection-over-Union (IoU) measures spatial overlap between candidate proposal $P$ and ground-truth box $G$:
$$\text{IoU}(P, G) = \frac{\text{Area}(P \cap G)}{\text{Area}(P \cup G)}$$
- **Positive Foreground Samples ($u \ge 1$):** Proposals with $\text{IoU} \ge 0.50$ assigned the matching ground-truth class.
- **Negative Background Samples ($u = 0$):** Proposals with $\text{IoU} < 0.15$ assigned the `Background` class.

### Step 3: Region Warping & Deep CNN Feature Extraction
- Each candidate proposal is cropped from the input image and warped via affine scaling into a fixed **$224 \times 224$** canvas.
- Passed through a convolutional backbone (ResNet18) to extract a 512-dimensional semantic feature vector.

### Step 4: Multi-Task Training (Classification + BBox Regression)
The network optimizes a joint multi-task loss function:
$$\mathcal{L} = \mathcal{L}_{\text{cls}}(p, u) + \lambda [u \ge 1] \mathcal{L}_{\text{reg}}(t, v)$$
Where:
- $\mathcal{L}_{\text{cls}}$ is Cross-Entropy classification loss across 4 classes (`Background`, `Car`, `Pedestrian`, `Traffic_Sign`).
- $\mathcal{L}_{\text{reg}}$ is Smooth L1 regression loss on bounding box parameterized deltas:
  $$t_x = (g_x - p_x) / p_w, \quad t_y = (g_y - p_y) / p_h$$
  $$t_w = \log(g_w / p_w), \quad t_h = \log(g_h / p_h)$$

### Step 5: Inference & Non-Maximum Suppression (NMS)
1. Selective search extracts candidate regions from an unseen test image.
2. Warped candidates are classified, and low-confidence proposals ($< 0.55$) or background classifications are filtered.
3. Bounding box regression shifts candidate proposals towards the true object boundary.
4. **Non-Maximum Suppression (NMS, $\theta_{\text{IoU}} = 0.25$)** eliminates duplicate overlapping boxes, selecting the highest-confidence bounding box.

---

## Architectural Comparison: CNN vs. R-CNN vs. Fast R-CNN vs. Faster R-CNN

| Feature / Model | Standard CNN | R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) |
| :--- | :--- | :--- | :--- | :--- |
| **Region Proposal** | Sliding Window (Dense) | Selective Search (~2k) | Selective Search (~2k) | **RPN (Region Proposal Net)** |
| **Feature Extraction** | Per-window CNN pass | ~2,000 passes / image | **Single pass on image** | **Single pass on image** |
| **RoI Processing** | Crop & resize before CNN | Warp to $224 \times 224$ | **RoI Pooling layer** | **RoI Pooling / RoIAlign** |
| **Inference Time** | > 150 seconds | 40 - 50 seconds | 2.0 seconds | **0.20 seconds (5 - 10 FPS)** |
| **Training Scheme** | Multi-stage classification | 3-stage (CNN, SVM, Reg) | Single-stage multi-task | **Fully end-to-end differentiable** |
| **Key Limitation** | Huge compute redundancy | Extremely slow, high disk use | CPU Selective Search bottleneck | Two-stage architectural complexity |

---

## Directory & File Structure
```
lab8/
├── data/                                 # Traffic dataset
│   ├── train_images/                     # Training scenes
│   ├── test_images/                      # Unseen test scenes
│   └── annotations.csv                   # Ground-truth annotations
├── outputs/                              # Results and artifacts
│   ├── architectural_comparison_report.txt # Full textual comparison
│   ├── architectural_comparison_summary.png# Visual comparison chart & table
│   ├── data_exploration_gt.png           # Ground-truth annotations plot
│   ├── rcnn_detection_results.png        # Final R-CNN detection predictions
│   ├── rcnn_model.pth                    # Trained PyTorch model checkpoint
│   ├── rcnn_proposals_dataset.npz        # Preprocessed proposals and deltas
│   ├── rcnn_training_loss.png            # Multi-task loss & accuracy curves
│   └── selective_search_proposals.png    # Selective search proposals visualization
├── Lab08_Solutions.ipynb                 # Interactive Jupyter Notebook
├── task0_data_exploration.py             # Dataset exploration & GT inspection
├── task1_selective_search.py             # Selective Search & IoU matching
├── task2_rcnn_training.py                # R-CNN architecture & training
├── task3_rcnn_inference.py               # Test inference & NMS
├── task4_architectural_comparison.py     # Detailed R-CNN family analysis
├── generate_dataset.py                   # Traffic dataset generator
├── run_all.py                            # Master runner executing all tasks
└── README.md                             # Technical documentation
```

---

## How to Run

### Run Complete Suite:
```bash
python lab8/run_all.py
```

### Run Individual Modules:
```bash
# 1. Data exploration and ground-truth visualization
python lab8/task0_data_exploration.py

# 2. Selective Search and IoU proposal generation
python lab8/task1_selective_search.py

# 3. Train R-CNN model
python lab8/task2_rcnn_training.py

# 4. Evaluate inference on test images with NMS
python lab8/task3_rcnn_inference.py

# 5. Generate architectural comparison report & chart
python lab8/task4_architectural_comparison.py
```

---

## Experimental Results & Visualizations
- **Data Exploration (`data_exploration_gt.png`):** Accurate ground-truth bounding box overlay across classes with zero coordinate clipping.
- **Selective Search (`selective_search_proposals.png`):** Extracted proposals accurately envelope ground truth objects with IoU $\ge 0.50$.
- **Training Convergence (`rcnn_training_loss.png`):** Multi-task loss drops monotonically as classification accuracy reaches $> 95\%$.
- **Detection Results (`rcnn_detection_results.png`):** Unseen test scenes accurately identify and localize `Car`, `Pedestrian`, and `Traffic_Sign` instances with high confidence scores ($> 80\%$) and clean NMS box de-duplication.
