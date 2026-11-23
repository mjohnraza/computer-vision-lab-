# Computer Vision Lab 07: Geometric Transformations, Perspective Correction, Camera Calibration & 3D Reconstruction

**Institution:** National University of Computer & Emerging Sciences (FAST-NUCES), Karachi  
**Department:** Department of Artificial Intelligence  
**Course:** Computer Vision Lab  
**Student Name:** Muhammad John Raza  
**Roll Number:** 23k-0069  
**Instructor:** Talha Shahid  

---

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Question 1: Image Transformations](#question-1-image-transformations)
   - [Q1(a): 2D Transformations](#q1a-2d-transformations)
   - [Q1(b): 3D Transformations](#q1b-3d-transformations)
   - [Q1(c): Homogeneous Coordinates in Computer Vision](#q1c-homogeneous-coordinates-in-computer-vision)
   - [Q1(d): Affine Transformation & Line Straightness Invariance](#q1d-affine-transformation--line-straightness-invariance)
   - [Q1(e): Projective Transformation (Homography) on Slanted Book](#q1e-projective-transformation-homography-on-slanted-book)
4. [Question 2: Perspective Correction (Unwarping)](#question-2-perspective-correction-unwarping)
5. [Question 3: Undistort Live Camera Feed](#question-3-undistort-live-camera-feed)
6. [Question 4: Build a 3D Model from Images Using COLMAP](#question-4-build-a-3d-model-from-images-using-colmap)
7. [How to Run](#how-to-run)
8. [Summary of Generated Artifacts](#summary-of-generated-artifacts)

---

## Overview
This laboratory covers fundamental geometric and projective transformations, planar homography, camera calibration and lens distortion models, real-time video feed undistortion, and multi-view 3D scene reconstruction (Structure-from-Motion and Multi-View Stereo).

---

## Directory Structure
```
lab7/
├── assets/                               # Test and calibration datasets
│   ├── book_original_flat.jpg            # Fronto-parallel reference document
│   ├── book_slanted.jpg                  # Slanted book on textured desk
│   ├── chessboard_01.jpg .. 10.jpg       # Multi-angle 9x6 calibration checkerboards
│   └── multiview_01.jpg .. 08.jpg        # 8-view multi-angle sequence of 3D object
├── outputs/                              # Generated visualizations and models
│   ├── camera_calibration.npz            # Camera matrix K, dist coeffs, rvecs, tvecs
│   ├── camera_matrix.json                # Human-readable intrinsic calibration report
│   ├── distorted_synthetic_stream.mp4    # Distorted live stream feed
│   ├── q1a_2d_transformations.png         # 2D transformations comparison plot
│   ├── q1b_3d_transformations.png         # 3D Euler rotations and translation plot
│   ├── q1c_homogeneous_coordinates_explanation.txt # Formal proof and CV rationale
│   ├── q1d_affine_straight_lines.png      # Collinearity preservation verification
│   ├── q1e_homography_book_correction.png # Slanted book unwarping result
│   ├── q1e_unwarped_book.jpg             # High-res unwarped book image
│   ├── q2_perspective_correction.png     # Inverse homography document scanning
│   ├── q2_unwarped_document.jpg          # Rectified document
│   ├── q3_camera_calibration_undistort.png # Calibration & undistortion plot
│   ├── q3_side_by_side_snapshot.jpg      # Live distorted vs undistorted snapshot
│   ├── q3_undistorted_feed.mp4           # Processed real-time video stream
│   ├── q4_colmap_3d_reconstruction.png   # 4-panel 3D SfM reconstruction figure
│   ├── scene_dense.ply                   # Dense colored 3D point cloud
│   ├── scene_mesh.ply                    # 3D surface triangle mesh (Stanford PLY)
│   └── scene_mesh.obj                    # 3D surface triangle mesh (Wavefront OBJ)
├── Lab07_Solutions.ipynb                 # Fully executed Jupyter Notebook with inline outputs
├── task1_image_transformations.py        # Standalone Python script for Question 1
├── task2_perspective_correction.py       # Standalone Python script for Question 2
├── task3_camera_calibration_undistort.py # Standalone Python script for Question 3
├── task4_colmap_3d_reconstruction.py     # Standalone Python script for Question 4
├── generate_assets.py                    # Dataset and asset generator script
├── run_all.py                            # Master test runner script
└── README.md                             # Comprehensive technical documentation
```

---

## Question 1: Image Transformations

### Q1(a): 2D Transformations
We implemented the fundamental 2D transformations:
1. **Translation:** Shifts position by $(t_x, t_y)$ using $M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$.
2. **Rotation:** Counter-clockwise rotation about the image center by angle $\theta$ using `cv2.getRotationMatrix2D`.
3. **Scaling:** Non-uniform and uniform scaling by factors $(s_x, s_y)$.
4. **Shearing:** Proportional displacement along horizontal axis parallel to vertical coordinate ($x' = x + sh_x \cdot y$).
5. **Composite Affine:** Single matrix multiplication chaining translation, rotation, scale, and shear.

### Q1(b): 3D Transformations
Rigid body motion in $\mathbb{R}^3$ transforms object vertices via:
$$P' = R P + T$$
where $R = R_z(\gamma) R_y(\beta) R_x(\alpha) \in SO(3)$ and $T = [t_x, t_y, t_z]^T$. We visualized canonical wireframe models rotated along X (pitch), Y (yaw), and Z (roll), as well as translated along all three axes in 3D space.

### Q1(c): Homogeneous Coordinates in Computer Vision
**Representation:**  
A 2D point $(x, y)$ is represented in projective space $\mathbb{P}^2$ as:
$$\mathbf{X} = [x, y, 1]^T \sim [w x, w y, w]^T \quad (w \ne 0)$$
Cartesian coordinates are recovered via dehomogenization: $x = X/W, y = Y/W$.

**Why Homogeneous Coordinates are Essential in Computer Vision:**
1. **Linearization of Translations:** Translation is non-linear in Euclidean 2D ($x' = x + t_x$). By augmenting with $W=1$, translations become linear $3\times 3$ matrix multiplications.
2. **Matrix Composition:** Any chain of geometric operations collapses into a single $3\times 3$ matrix $M = T_n \cdots T_1$. Points are mapped via a single operation $p' = M p$.
3. **Representation of Points at Infinity & Vanishing Points:** Parallel lines intersect at points with $W = 0$, representing pure directions. Under central projection, these project to finite vanishing points.
4. **Pinhole Camera Central Perspective Projection:** Central projection maps 3D world coordinates to 2D image coordinates linearly:
   $$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
   The scale factor $\lambda$ represents projective depth ($Z$), directly modeling foreshortening.

### Q1(d): Affine Transformation & Line Straightness Invariance
- **Mathematical Proof:** A straight line in parametric form is $L(t) = P_0 + t \vec{v}$. An affine transformation is $T(X) = A X + b$. Applying $T$ yields $T(L(t)) = (A P_0 + b) + t (A \vec{v}) = P_0' + t \vec{v}'$, which is another straight line.
- **Empirical Verification:** 100 sample points along a diagonal line were transformed via an affine matrix and fitted with a straight line. Maximum perpendicular deviation was **$4.17 \times 10^{-6}$ pixels** (zero to machine precision), proving collinearity is preserved.

### Q1(e): Projective Transformation (Homography) on Slanted Book
Given 4 corner correspondences of a slanted book on a desk:
- Source quadrilateral: `[[260, 90], [820, 160], [710, 720], [130, 580]]`
- Destination rectangle: `[[0, 0], [599, 0], [599, 799], [0, 799]]`
- Computed 3x3 Homography $H$ using `cv2.getPerspectiveTransform` and warped via `cv2.warpPerspective` with Lanczos interpolation, producing a perfectly flat, fronto-parallel book page.

---

## Question 2: Perspective Correction (Unwarping)
Taking a photo of a flat surface captured at an angle and correcting it to appear head-on.

**Function:** `apply_perspective_transform(image, pts)`:
- Sorts corners into $[TL, TR, BR, BL]$ automatically using $(x+y)$ sum and $(y-x)$ difference.
- Dynamically computes target width $W = \max(\|BR - BL\|, \|TR - TL\|)$ and height $H = \max(\|TR - BR\|, \|TL - BL\|)$.
- Computes the **inverse homography matrix** mapping the tilted quadrilateral to the rectangular destination.
- Rectified canvas dimensions: **$596 \times 569$ pixels**.
- Output saved to `outputs/q2_perspective_correction.png` and `outputs/q2_unwarped_document.jpg`.

---

## Question 3: Undistort Live Camera Feed
**Lens Distortion Model:**
- Radial: $x_{\text{dist}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$
- Tangential: $x_{\text{dist}} = x + [2 p_1 x y + p_2 (r^2 + 2 x^2)]$

**Results:**
- Checkerboard calibration ($9 \times 6$ corners, 10 images).
- Overall RMS Reprojection Error: **0.0618 pixels**.
- Mean Reprojection Error: **0.0083 pixels**.
- Intrinsic Matrix $K$:
  $$K = \begin{bmatrix} 749.72 & 0 & 400.33 \\ 0 & 749.61 & 299.87 \\ 0 & 0 & 1 \end{bmatrix}$$
- Real-time video stream capture using `cv2.VideoCapture()`.
- Undistorted each frame using `cv2.undistort()` with `getOptimalNewCameraMatrix`.
- Processed 108 frames with real-time FPS display and side-by-side video recorded to `outputs/q3_undistorted_feed.mp4`.

---

## Question 4: Build a 3D Model from Images Using COLMAP
Automated 3D reconstruction pipeline from multi-view photographs:
1. **Feature Detection:** SIFT keypoints invariant to scale, rotation, and illumination (~1800 keypoints per view).
2. **Feature Matching:** Lowe's ratio test ($0.75$) with geometric RANSAC inlier verification.
3. **Camera Pose Estimation:** Essential matrix $E = K^T F K$ decomposed into rotation $R$ and translation $t$ via `cv2.recoverPose`.
4. **3D Triangulation:** Multi-view triangulation of matched rays into metric 3D point cloud coordinates.
5. **Dense Point Cloud Construction:** 4,754 colored 3D points saved to `outputs/scene_dense.ply`.
6. **3D Surface Meshing:** Open3D surface reconstruction (Ball-Pivoting & Poisson) generating **6,594 triangle faces**, saved to `outputs/scene_mesh.ply` and `outputs/scene_mesh.obj`.
7. **COLMAP Automation Wrapper:** CLI wrapper automating `colmap feature_extractor`, `exhaustive_matcher`, `mapper`, `image_undistorter`, `patch_match_mvs`, `stereo_fusion`, and `poisson_mesher`.

---

## How to Run

### Run Complete Suite:
```bash
python lab7/run_all.py
```

### Run Individual Tasks:
```bash
# Task 1: Image Transformations (2D, 3D, Homogeneous, Affine, Homography)
python lab7/task1_image_transformations.py

# Task 2: Perspective Correction (Unwarping)
python lab7/task2_perspective_correction.py

# Task 3: Camera Calibration & Live Feed Undistortion
python lab7/task3_camera_calibration_undistort.py

# Task 4: 3D Model Reconstruction (COLMAP / SfM)
python lab7/task4_colmap_3d_reconstruction.py
```

### Run Jupyter Notebook:
Open `lab7/Lab07_Solutions.ipynb` in VS Code or JupyterLab and run all cells.

---

## Summary of Generated Artifacts
| File | Description | Status |
|------|-------------|--------|
| `Lab07_Solutions.ipynb` | Complete interactive notebook with all executed outputs | Generated & Executed |
| `task1_image_transformations.py` | 2D/3D transformations, homogeneous coordinates, affine straightness | Verified |
| `task2_perspective_correction.py` | Perspective rectification & inverse homography | Verified |
| `task3_camera_calibration_undistort.py` | Camera calibration & real-time undistortion feed | Verified |
| `task4_colmap_3d_reconstruction.py` | COLMAP CLI wrapper & Open3D 3D reconstruction engine | Verified |
| `camera_calibration.npz` | Intrinsic matrix $K$ and distortion coefficients $D$ | Verified |
| `camera_matrix.json` | Calibration report and parameters | Verified |
| `scene_dense.ply` | Dense colored 3D point cloud | Verified |
| `scene_mesh.ply` / `.obj` | Reconstructed 3D surface mesh (6594 triangles) | Verified |
| `q3_undistorted_feed.mp4` | Live side-by-side undistorted video stream | Verified |
