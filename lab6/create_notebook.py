"""
Generates the complete Jupyter Notebook Lab06_Solutions.ipynb for submission.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# FAST NUCES - Computer Vision (CS4063)
# Lab 06: Edges, Lines, Circles, Keypoints, Wavelets
**Instructor:** Talha Shahid  
**Topic:** Structural Feature Extraction, Geometric Transforms, SIFT, and Wavelet Multiresolution Analysis  

---

## Lab Objectives & Rules
- Implement **8 complete computer vision scenarios**:
  1. Computer Screen Detection (Hough Lines)
  2. Asset Tracking in a Computer Lab (SIFT)
  3. Anomaly Detection in Sensor Data (Wavelet Analysis)
  4. Object Recognition in Video (SIFT + Homography)
  5. Panoramic Image Stitching (SIFT + Perspective Warping)
  6. Autonomous Vehicle Lane Detection (Hough Lines)
  7. Coins Detection and Automated Counting (Hough Circles)
  8. Smart Security System (Zone Boundary Detection + Baseline Calibration)
  + **Foundation Section**: Sobel, Canny, and Laplacian of Gaussian (LoG)
- For every task:
  - Display the original input and resulting output.
  - State every parameter chosen and a concrete line of why.
  - Save output figures to disk (`plt.savefig` or `cv2.imwrite`).
"""))

# Imports & Setup
cells.append(nbf.v4.new_markdown_cell("""## 0. Setup and Environment Initialization
Ensure all required libraries are loaded: OpenCV, NumPy, Matplotlib, PyWavelets (`pywt`), and Pandas.
"""))

cells.append(nbf.v4.new_code_cell("""import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import os

# Ensure assets directory exists and generate test images if needed
import generate_all_assets
generate_all_assets.make_edges_input()
generate_all_assets.make_lab_screens()
generate_all_assets.make_asset_tracking_data()
generate_all_assets.make_video_object_data()
generate_all_assets.make_panorama_data()
generate_all_assets.make_road_data()
generate_all_assets.make_security_video()

print(f"OpenCV Version: {cv2.__version__}")
print(f"PyWavelets Version: {pywt.__version__}")
"""))

# Foundation: Boundary Detection
cells.append(nbf.v4.new_markdown_cell("""---
## Foundation: Boundary Detection (Sobel, Canny, LoG)

### Theory & Mathematical Intuition:
1. **Sobel First Derivatives**:
   - $G_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix} * I$, $G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix} * I$
   - Uses `cv2.CV_64F` because derivatives take negative values on bright-to-dark edges.
   - Magnitude: $G = \\sqrt{G_x^2 + G_y^2}$.
2. **Canny Edge Detector**:
   - Gaussian smoothing $\\to$ Gradient computation $\\to$ Non-maximum suppression (NMS) $\\to$ Hysteresis thresholding ($50, 150$).
3. **Laplacian of Gaussian (LoG)**:
   - Second derivative $\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$. Edges lie at zero-crossings.
"""))

cells.append(nbf.v4.new_code_cell("""# Foundation: Sobel, Canny, LoG
import task0_edges
task0_edges.run_task0("assets/edges_input.jpg", "task0_edges_output.png")

# Display inline
res_img = cv2.imread("task0_edges_output.png")
plt.figure(figsize=(15, 9))
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Foundation: Boundary Detection Comparison (Sobel, Canny, LoG)", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 1: Screen Detection
cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Computer Screen Detection in a Computer Lab
**Technique:** Hough Line Transformation (`HoughLinesP`), Morphological Closing, Contour Analysis, and Luminance Profiling.

### Parameters Chosen & Justification:
- `Canny(50, 150)`: 1:3 ratio isolates outer bezel borders while suppressing interior display reflections.
- `HoughLinesP(threshold=25, minLineLength=25, maxLineGap=25)`:
  - `threshold=25`: Minimum votes to qualify line segment.
  - `minLineLength=25`: Ignores small noise edges and desk grain.
  - `maxLineGap=25`: Bridges corners between monitor bezels.
- `Morphological Close (9x9)`: Connects perpendicular segments into closed bounding boxes.
- `Contour Area > 8000 & 1.1 < Aspect Ratio < 2.0`: Standard widescreen monitor dimensions.
- `Mean Luminance > 90`: Distinguishes active glowing screens ('ON') from turned-off black panels ('OFF').
"""))

cells.append(nbf.v4.new_code_cell("""import task1_screen_detection
screens = task1_screen_detection.detect_screens("assets/lab_screens.jpg", "task1_screens_output.png", expected_count=6)

plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(cv2.imread("task1_screens_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 1: Computer Screen Detection & Status Monitoring", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 2: Asset Tracking
cells.append(nbf.v4.new_markdown_cell("""---
## Task 2: Asset Tracking in a Computer Lab Using SIFT
**Technique:** SIFT (Scale-Invariant Feature Transform) + Brute-Force Matcher + Lowe's Ratio Test + RANSAC Homography.

### Parameters Chosen & Justification:
- `cv2.SIFT_create()`: Extracts 128-dimensional descriptors invariant to scale, rotation, and illumination.
- `Lowe's Ratio Test = 0.75`: Requires the best match distance to be $< 0.75 \\times$ the runner-up distance, rejecting ambiguous descriptors.
- `RANSAC Reprojection Error = 5.0 pixels`: Tolerates minor perspective distortions while eliminating spatial outliers.
- `min_inliers = 15`: Ensures high-confidence certification of asset presence before logging in the inventory.
"""))

cells.append(nbf.v4.new_code_cell("""import task2_asset_tracking
inventory = task2_asset_tracking.track_assets("assets/lab_scene.jpg", "task2_asset_tracking_output.png", min_inliers=15)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(cv2.imread("task2_asset_tracking_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 2: SIFT Asset Tracking & Inventory Verification", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 3: Wavelet Anomaly Detection
cells.append(nbf.v4.new_markdown_cell("""---
## Task 3: Anomaly Detection in Sensor Data Using Wavelet Transformation
**Technique:** Discrete Wavelet Transform (DWT), Noise Estimation via MAD, VisuShrink Universal Soft Thresholding, Inverse DWT, and Z-Score Outlier Flagging.

### Parameters Chosen & Justification:
- `Wavelet = 'db4'`: Daubechies 4 wavelet provides compact support and smooth wavelets, avoiding Haar blockiness.
- `Decomposition Level = 4`: Separates low-frequency operational dynamics from high-frequency shock faults.
- `Noise Estimation via MAD`: $\\hat{\\sigma} = \\frac{\\text{median}(|cD_1|)}{0.6745}$. Resistant to isolated impulse outliers.
- `Universal Threshold`: $\\lambda = \\hat{\\sigma} \\sqrt{2 \\ln(N)}$. Asymptotically suppresses Gaussian noise floor.
- `Z-Score Cutoff > 3.0`: Standard 3-sigma empirical threshold (identifies samples outside $99.73\\%$ normal residual variation).
"""))

cells.append(nbf.v4.new_code_cell("""import task3_wavelet_anomaly
anom_res = task3_wavelet_anomaly.run_wavelet_anomaly_detection("assets/sensor_data.csv", "task3_wavelet_anomaly.png")

plt.figure(figsize=(13, 8))
plt.imshow(cv2.cvtColor(cv2.imread("task3_wavelet_anomaly.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 3: Sensor Anomaly Detection Matching Lab Sheet Format", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 4: Video Object Recognition
cells.append(nbf.v4.new_markdown_cell("""---
## Task 4: Object Recognition (Using Video)
**Technique:** SIFT Feature Tracking across frames + RANSAC Homography + `cv2.perspectiveTransform` Bounding Box Localization.

### Parameters Chosen & Justification:
- `Lowe's Ratio Test = 0.75`: Discards false descriptor matches under scale and perspective transformations.
- `RANSAC Reprojection Threshold = 5.0`: Computes the projective homography matrix $H$ mapping reference coordinates to current video frame.
- `min_inliers = 15`: High confidence threshold prevents false-positive bounding boxes during object absence.
"""))

cells.append(nbf.v4.new_code_cell("""import task4_object_recognition
recog_res = task4_object_recognition.run_object_recognition("assets/object.jpg", "assets/test_video.mp4")

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(cv2.imread("task4_recognition_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 4: SIFT Object Recognition & Frame-by-Frame Tracking", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 5: Panorama Stitching
cells.append(nbf.v4.new_markdown_cell("""---
## Task 5: Your Panoramic Image (SIFT Stitching)
**Technique:** SIFT Feature Detection & Matching + Pairwise RANSAC Homography + Perspective Warping + Seamless Compositing.

### Parameters Chosen & Justification:
- `cv2.BFMatcher(cv2.NORM_L2)`: Matches 128-d Euclidean descriptor vectors between overlapping panning photos.
- `Ratio 0.75`: Effectively identifies true correspondence pairs across overlapping fields of view.
- `Homography Mapping`: Transforms right image coordinates onto the left reference image plane.
- `Non-destructive Blending`: Merges overlapping regions without black border occlusion.
"""))

cells.append(nbf.v4.new_code_cell("""import task5_panorama
pano = task5_panorama.create_panorama(["assets/pan1.jpg", "assets/pan2.jpg", "assets/pan3.jpg"])

plt.figure(figsize=(16, 12))
plt.imshow(cv2.cvtColor(cv2.imread("task5_panorama_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 5: Multi-Image Panning Alignment and Panoramic Stitching", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 6: Lane Detection
cells.append(nbf.v4.new_markdown_cell("""---
## Task 6: Autonomous Vehicle Lane Detection
**Technique:** Gaussian Blur + Canny Edge Detection + Trapezoidal ROI Mask + Probabilistic Hough Lines (`HoughLinesP`) + Slope Polarity Separation + Corridor Fill.

### Parameters Chosen & Justification:
- `Trapezoidal ROI`: Restricts detection to the vehicle's forward road surface, ignoring horizon, sky, and roadside foliage.
- `HoughLinesP(threshold=25, minLineLength=25, maxLineGap=80)`:
  - `maxLineGap=80`: Crucial for connecting dashed road markings into continuous trajectories.
  - `minLineLength=25`: Filters out gravel, asphalt texture, and minor pavement imperfections.
- `Slope Polarity Separation`:
  - In image coordinates ($y$ increases downwards), Left Lane has $m < 0$, Right Lane has $m > 0$.
- `Lane Extrapolation`: Fits median slopes and intercepts from hood level ($y=h$) to horizon ($y=0.58h$).
"""))

cells.append(nbf.v4.new_code_cell("""import task6_lane_detection
lane_res = task6_lane_detection.detect_lanes("assets/road.jpg", "task6_lane_output.png")

plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(cv2.imread("task6_lane_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 6: Autonomous Vehicle Lane Detection and Corridor Overlay", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 7: Coins Counting
cells.append(nbf.v4.new_markdown_cell("""---
## Task 7: Coins Detection and Automated Counting
**Technique:** Median Blur + Hough Gradient Circle Transformation (`cv2.HoughCircles`) + Centroid Marking + Sequential Coin Labeling.

### Parameters Chosen & Justification:
- `medianBlur(gray, 5)`: Suppresses specular coin glare and metal texture noise without blurring boundary circular edges.
- `HOUGH_GRADIENT with dp=1.2`: Gradient direction vectors vote for center coordinates in 2D space, eliminating 3D cone search cost.
- `minDist=28`: Enforces physical minimum spacing between coin centers, preventing duplicate concentric detections on the same coin.
- `param1=50`: Upper Canny threshold inside the circle detector.
- `param2=30`: Accumulator vote threshold for accepting a circle center.
- `minRadius=12, maxRadius=38`: Constrains search space strictly to physical coin sizes.
"""))

cells.append(nbf.v4.new_code_cell("""import task7_coins_counting
coin_res = task7_coins_counting.count_coins("coins.jpg", "task7_coins_output.png")

plt.figure(figsize=(15, 6))
plt.imshow(cv2.cvtColor(cv2.imread("task7_coins_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 7: Hough Circle Coin Detection & Sequential Counting", fontsize=14, fontweight="bold")
plt.show()
"""))

# Task 8: Smart Security System
cells.append(nbf.v4.new_markdown_cell("""---
## Task 8: Smart Security System Using Boundary Detection
**Technique:** Polygonal Zone Masking + Canny Boundary Detection + Statistical Baseline Calibration + Temporal Streak Filtering.

### Parameters Chosen & Justification:
- `Predefined Security Polygon`: High-value restricted zone coordinates `[(200, 150), (500, 150), (500, 400), (200, 400)]`.
- `Baseline Calibration Window = 30 frames`: Measures ambient background noise ($\mu_{\text{base}}, \sigma_{\text{base}}$) in the empty zone.
- `Alarm Trigger Threshold`: $T = \mu_{\text{base}} + 4\sigma_{\text{base}} + 50$ edge pixels. Exceeds $99.99\%$ of ambient noise variations.
- `Streak Filter = 4 consecutive frames`: Prevents transient camera noise, single-frame lighting flicker, or small shadows from causing false alarms.
"""))

cells.append(nbf.v4.new_code_cell("""import task8_smart_security
sec_res = task8_smart_security.run_security_system("assets/security_video.mp4", "task8_security_alarm.mp4", "task8_security_output.png")

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(cv2.imread("task8_security_output.png"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Task 8: Smart Security Boundary Detection & Intrusion Alarm Timeline", fontsize=14, fontweight="bold")
plt.show()
"""))

# Summary & Recall
cells.append(nbf.v4.new_markdown_cell("""---
## Summary of Lab 06 Results & Method Comparison

| Method | Task | Question Answered | Key Parameter(s) | Main Failure Mode & Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Foundation | Gradient magnitude & direction | `ksize=3`, `cv2.CV_64F` | Noise sensitivity; mitigate with pre-blurring. |
| **Canny** | Foundation / 1, 6, 8 | Thin connected edges | Blur $\\sigma$, thresholds $(50, 150)$ | Over-segmentation / noise; mitigate with hysteresis. |
| **LoG** | Foundation | Inflection point / zero-crossing | Blur $\\sigma$, `cv2.Laplacian` | Extreme noise sensitivity; require Gaussian pre-smoothing. |
| **HoughLinesP** | Task 1 & 6 | Finite straight line segments | `threshold`, `minLineLength`, `maxLineGap` | Disconnected dashes; bridge with `maxLineGap`. |
| **HoughCircles** | Task 7 | Circular object centers & radii | `minDist`, `param2`, radius range | Duplicate detections; enforce `minDist` between coins. |
| **SIFT** | Task 2, 4, 5 | Scale & rotation invariant keypoints | Ratio $=0.75$, RANSAC reproj $=5.0$ | Textureless / repetitive surfaces; filter with Lowe's ratio. |
| **Wavelets** | Task 3 | Transient time-frequency anomalies | Wavelet family, level, MAD threshold | Coarse wavelet selection; use `db4` with VisuShrink. |

---
**End of Lab 06 Notebook**
"""))

nb.cells = cells

with open("Lab06_Solutions.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Lab06_Solutions.ipynb successfully created!")
