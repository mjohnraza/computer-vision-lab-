"""
Task 1: Image Transformations (2D, 3D, Homogeneous Coordinates, Affine Straightness, Homography)
==================================================================================================
Author: Muhammad John Raza
Roll No: 23k-0069
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 07)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    if rel_path.startswith("lab7/") and os.path.exists(rel_path[5:]):
        return rel_path[5:]
    if os.path.exists(os.path.join("lab7", rel_path)):
        return os.path.join("lab7", rel_path)
    if os.path.basename(os.getcwd()) == "lab7" and rel_path.startswith("lab7/"):
        return rel_path[5:]
    return rel_path

output_dir = "outputs" if os.path.basename(os.getcwd()) == "lab7" else "lab7/outputs"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# Q1(a): 2D Transformations (Translation, Rotation, Scaling, Shearing)
# =====================================================================
def run_q1a_2d_transformations(input_path="lab7/assets/book_original_flat.jpg"):
    print("\n--- Running Q1(a): 2D Image Transformations ---")
    real_path = get_path(input_path)
    img = cv2.imread(real_path)
    if img is None:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (80, 80), (320, 320), (50, 150, 240), -1)
        cv2.circle(img, (200, 200), 70, (255, 255, 255), -1)
        cv2.putText(img, "TEST 2D", (130, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    h, w = img.shape[:2]

    # 1. Translation: Tx = 70, Ty = 50
    tx, ty = 70, 50
    M_trans = np.float32([[1, 0, tx],
                          [0, 1, ty]])
    img_trans = cv2.warpAffine(img, M_trans, (w, h), borderValue=(240, 240, 240))

    # 2. Rotation: 35 degrees counter-clockwise about image center
    center = (w // 2, h // 2)
    angle = 35.0
    scale = 1.0
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    img_rot = cv2.warpAffine(img, M_rot, (w, h), borderValue=(240, 240, 240))

    # 3. Scaling: Non-uniform scaling (sx = 0.65, sy = 0.85) centered
    sx, sy = 0.65, 0.85
    M_scale = np.float32([[sx, 0, (1 - sx) * (w / 2)],
                          [0, sy, (1 - sy) * (h / 2)]])
    img_scale = cv2.warpAffine(img, M_scale, (w, h), borderValue=(240, 240, 240))

    # 4. Shearing: Horizontal shear sh_x = 0.35, centered
    sh_x = 0.35
    M_shear = np.float32([[1, sh_x, -sh_x * (h / 2)],
                          [0, 1,    0]])
    img_shear = cv2.warpAffine(img, M_shear, (w, h), borderValue=(240, 240, 240))

    # 5. Combined Affine: Translation + Rotation + Scaling + Shearing
    M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
    M_scale_3x3 = np.vstack([M_scale, [0, 0, 1]])
    M_shear_3x3 = np.vstack([M_shear, [0, 0, 1]])
    M_trans_3x3 = np.vstack([M_trans, [0, 0, 1]])
    M_combined = (M_trans_3x3 @ M_rot_3x3 @ M_shear_3x3 @ M_scale_3x3)[:2]
    img_combined = cv2.warpAffine(img, M_combined, (w, h), borderValue=(240, 240, 240))

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = [
        "Original Image",
        f"1. Translation (tx={tx}, ty={ty})",
        f"2. Rotation ({angle} deg, Center)",
        f"3. Scaling (sx={sx}, sy={sy})",
        f"4. Shearing (sh_x={sh_x})",
        "5. Combined 2D Transformation"
    ]
    images = [img, img_trans, img_rot, img_scale, img_shear, img_combined]

    for ax, im, title in zip(axes.flat, images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.axis('on')
        ax.grid(color='gray', linestyle=':', alpha=0.5)

    plt.suptitle("Q1(a): 2D Image Transformations Analysis\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q1a_2d_transformations.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved 2D transformation visualization: {output_fig}")
    return output_fig


# =====================================================================
# Q1(b): 3D Transformations (Object rotation & translation along X, Y, Z)
# =====================================================================
def run_q1b_3d_transformations():
    print("\n--- Running Q1(b): 3D Object Transformations ---")
    vertices = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        [ 0, -1,  2], [ 0,  1,  2]
    ], dtype=np.float64)

    faces = [
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [4, 5, 8],
        [7, 6, 9],
        [4, 8, 9, 7],
        [5, 8, 9, 6]
    ]

    def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
        ax, ay, az = np.radians([rx_deg, ry_deg, rz_deg])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax),  np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az),  np.cos(az), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    scenarios = [
        ("1. Canonical 3D Object", 0, 0, 0, [0, 0, 0]),
        ("2. Rotation around X (Pitch +35 deg)", 35, 0, 0, [0, 0, 0]),
        ("3. Rotation around Y (Yaw +45 deg)", 0, 45, 0, [0, 0, 0]),
        ("4. Rotation around Z (Roll +50 deg)", 0, 0, 50, [0, 0, 0]),
        ("5. Move along X, Y, Z (T=[1.5, -1.0, 1.2])", 0, 0, 0, [1.5, -1.0, 1.2]),
        ("6. Full 3D Pose: R(30, 45, 25) + T", 30, 45, 25, [1.2, 0.8, -0.5])
    ]

    fig = plt.figure(figsize=(18, 11))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (title, rx, ry, rz, T) in enumerate(scenarios):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        R = get_rotation_matrix(rx, ry, rz)
        T_vec = np.array(T)
        transformed_vertices = (R @ vertices.T).T + T_vec

        poly_list = []
        for face in faces:
            poly_list.append([transformed_vertices[idx] for idx in face])

        poly_coll = Poly3DCollection(poly_list, alpha=0.65, edgecolor='black', linewidths=1.2)
        poly_coll.set_facecolor(colors[i])
        ax.add_collection3d(poly_coll)

        ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', arrow_length_ratio=0.15)
        ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', arrow_length_ratio=0.15)
        ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', arrow_length_ratio=0.15)

        ax.set_xlim([-3.0, 3.5])
        ax.set_ylim([-3.0, 3.5])
        ax.set_zlim([-3.0, 3.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.view_init(elev=22, azim=-55)

    plt.suptitle("Q1(b): 3D Rigid Transformations (Euler Rotations & Translations in R^3)\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q1b_3d_transformations.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved 3D transformation visualization: {output_fig}")
    return output_fig


# =====================================================================
# Q1(c): Homogeneous Coordinates Representation & Comprehensive Rationale
# =====================================================================
def run_q1c_homogeneous_coordinates():
    print("\n--- Running Q1(c): Homogeneous Coordinates Analysis ---")
    x, y = 145.0, 280.0
    P_homo = np.array([x, y, 1.0], dtype=np.float64)

    w_scale = 3.5
    P_scaled_homo = np.array([w_scale * x, w_scale * y, w_scale], dtype=np.float64)
    recovered_pt = (P_scaled_homo[:2] / P_scaled_homo[2])

    T1 = np.array([[1, 0, 50],
                   [0, 1, -30],
                   [0, 0, 1]], dtype=np.float64)
    th = np.radians(45)
    R2 = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th),  np.cos(th), 0],
                   [0,           0,          1]], dtype=np.float64)
    S3 = np.array([[1.5, 0,   0],
                   [0,   0.8, 0],
                   [0,   0,   1]], dtype=np.float64)

    M_composite = S3 @ R2 @ T1
    P_transformed_homo = M_composite @ P_homo
    P_transformed_euclid = P_transformed_homo[:2] / P_transformed_homo[2]

    l1 = np.array([-2, 1, -10], dtype=np.float64)
    l2 = np.array([-2, 1, -40], dtype=np.float64)
    p_inf = np.cross(l1, l2)

    explanation_text = f"""================================================================================
Q1(c) HOMOGENEOUS COORDINATES IN COMPUTER VISION: FORMAL EXPLANATION
================================================================================
1. REPRESENTATION:
   - Euclidean 2D Point: (x, y) = ({x:.1f}, {y:.1f})
   - Canonical Homogeneous Coordinate: [x, y, 1]^T = [{P_homo[0]:.1f}, {P_homo[1]:.1f}, {P_homo[2]:.1f}]^T
   - Equivalence under scale lambda={w_scale}:
     [{P_scaled_homo[0]:.1f}, {P_scaled_homo[1]:.1f}, {P_scaled_homo[2]:.1f}]^T
     Dehomogenization (divide by W={P_scaled_homo[2]}): ({recovered_pt[0]:.1f}, {recovered_pt[1]:.1f}) -> Exactly matches original!

2. WHY THE EXTRA COORDINATE (W) IS USED IN COMPUTER VISION:
   -----------------------------------------------------------------------------
   (a) LINEARIZATION OF AFFINE & PROJECTIVE TRANSFORMATIONS:
       In 2D Cartesian coordinates, translation is non-linear: x' = x + tx (addition).
       Rotation and scaling are linear: x' = Ax (multiplication).
       Affine transformations therefore require a hybrid equation: x' = Ax + t.
       By embedding 2D points into projective space P^2 with homogeneous coordinates,
       translation becomes a standard matrix multiplication:
         [x']   [a11 a12 tx] [x]
         [y'] = [a21 a22 ty] [y]
         [ 1]   [ 0   0   1] [1]
       This converts all geometric transformations into unified matrix operations.

   (b) EFFICIENT COMPOSITION OF MULTIPLE TRANSFORMATIONS:
       Because every transformation is a 3x3 matrix, any sequence of N operations
       can be pre-multiplied into a single 3x3 matrix: M = T_n * ... * T_2 * T_1.
       Every scene point is transformed with a single matrix-vector product p' = M * p.
       Calculated composite output for ({x:.1f}, {y:.1f}): ({P_transformed_euclid[0]:.2f}, {P_transformed_euclid[1]:.2f}).

   (c) REPRESENTATION OF POINTS AT INFINITY (IDEAL POINTS) & VANISHING POINTS:
       In projective space P^2, parallel lines intersect at a point at infinity:
         Line 1: -2x + y - 10 = 0 -> l1 = [-2, 1, -10]
         Line 2: -2x + y - 40 = 0 -> l2 = [-2, 1, -40]
         Intersection = l1 x l2 = [{p_inf[0]:.1f}, {p_inf[1]:.1f}, {p_inf[2]:.1f}]^T
       Notice W = 0! The point at infinity represents direction without division by zero.
       Under perspective projection (cameras), points at infinity map to finite
       vanishing points on the image plane (horizon line).

   (d) PINHOLE CAMERA PERSPECTIVE PROJECTION:
       The 3D to 2D perspective projection through a camera center is inherently projective:
         lambda * [u, v, 1]^T = K * [R | t] * [X_w, Y_w, Z_w, 1]^T
       Division by the scale factor lambda (depth Z) models physical perspective foreshortening.
================================================================================"""
    print(explanation_text)

    out_txt = os.path.join(output_dir, "q1c_homogeneous_coordinates_explanation.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(explanation_text)
    print(f"  -> Saved explanation: {out_txt}")


# =====================================================================
# Q1(d): Affine Transformation and Line Straightness Invariance
# =====================================================================
def run_q1d_affine_straight_lines():
    print("\n--- Running Q1(d): Affine Transformation & Line Straightness Analysis ---")
    size = 500
    grid_img = np.full((size, size, 3), 255, dtype=np.uint8)

    for i in range(50, size, 50):
        cv2.line(grid_img, (i, 0), (i, size), (220, 220, 220), 1)
        cv2.line(grid_img, (0, i), (size, i), (220, 220, 220), 1)

    cv2.line(grid_img, (50, 80), (450, 80), (255, 0, 0), 3)
    cv2.line(grid_img, (80, 50), (80, 450), (0, 180, 0), 3)
    cv2.line(grid_img, (50, 50), (450, 450), (0, 0, 255), 3)
    cv2.line(grid_img, (50, 450), (450, 50), (200, 0, 200), 3)
    cv2.putText(grid_img, "Collinear Points Test", (130, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

    src_tri = np.float32([[50, 50], [450, 50], [50, 450]])
    dst_tri = np.float32([[100, 70], [420, 130], [60, 430]])
    M_affine = cv2.getAffineTransform(src_tri, dst_tri)

    warped_grid = cv2.warpAffine(grid_img, M_affine, (size, size), borderValue=(255, 255, 255))

    P0 = np.array([50.0, 50.0])
    P1 = np.array([450.0, 450.0])
    t_vals = np.linspace(0, 1, 100)
    line_pts = np.array([P0 + t * (P1 - P0) for t in t_vals])

    ones = np.ones((len(line_pts), 1))
    pts_homo = np.hstack([line_pts, ones])
    transformed_pts = (M_affine @ pts_homo.T).T

    vx, vy, x0, y0 = cv2.fitLine(transformed_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    normal = np.array([-vy[0], vx[0]])
    distances = np.abs(np.dot(transformed_pts - np.array([x0[0], y0[0]]), normal))
    max_deviation = np.max(distances)

    print(f"  -> Quantitative Linearity Check:")
    print(f"     Maximum deviation of transformed points from fitted straight line: {max_deviation:.8f} pixels")
    print(f"     => Line remains perfectly straight! (Deviation = 0 to machine precision)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Grid & Straight Lines\n(Horizontal, Vertical, Diagonals)", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].imshow(cv2.cvtColor(warped_grid, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"After Affine Transformation\n(Lines remain perfectly straight, Max Error: {max_deviation:.2e} px)",
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Q1(d): Invariance of Straight Lines Under Affine Transformation\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q1d_affine_straight_lines.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved affine straight lines verification: {output_fig}")
    return output_fig


# =====================================================================
# Q1(e): Projective Transformation (Homography) on Slanted Book
# =====================================================================
def run_q1e_homography_perspective(slanted_path="lab7/assets/book_slanted.jpg"):
    print("\n--- Running Q1(e): Projective Transformation (Homography) on Slanted Book ---")
    real_path = get_path(slanted_path)
    img = cv2.imread(real_path)
    if img is None:
        raise FileNotFoundError(f"Missing input image at {slanted_path} (resolved {real_path})")

    src_pts = np.float32([
        [260, 90],
        [820, 160],
        [710, 720],
        [130, 580]
    ])

    target_w, target_h = 600, 800
    dst_pts = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print(f"  -> Computed 3x3 Homography Matrix H:\n{np.array2string(H, precision=4, suppress_small=True)}")

    unwarped_book = cv2.warpPerspective(img, H, (target_w, target_h), flags=cv2.INTER_LANCZOS4)

    annotated_orig = img.copy()
    pts_int = src_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated_orig, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    corner_labels = ["TL (260, 90)", "TR (820, 160)", "BR (710, 720)", "BL (130, 580)"]
    for pt, label in zip(src_pts, corner_labels):
        cv2.circle(annotated_orig, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(annotated_orig, label, (int(pt[0]) - 30, int(pt[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Slanted Image with Quadrilateral ROI\n(Perspective Distortion on Desk)",
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(unwarped_book, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Rectified Fronto-Parallel Book\n(Homography Perspective Correction)",
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Q1(e): Projective Transformation (Homography) Perspective Rectification\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q1e_homography_book_correction.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()

    cv2.imwrite(os.path.join(output_dir, "q1e_unwarped_book.jpg"), unwarped_book)
    print(f"  -> Saved rectified book output image and figure: {output_fig}")
    return output_fig


def main():
    print("=================================================================")
    print("      LAB 07 - TASK 1: IMAGE TRANSFORMATIONS SUITE              ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")
    run_q1a_2d_transformations()
    run_q1b_3d_transformations()
    run_q1c_homogeneous_coordinates()
    run_q1d_affine_straight_lines()
    run_q1e_homography_perspective()
    print("\n[SUCCESS] Task 1 completed successfully.")

if __name__ == "__main__":
    main()
