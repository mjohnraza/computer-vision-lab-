"""
Asset Generator for Lab 07: Geometric Transformations, Calibration & 3D Reconstruction
Author: Muhammad John Raza (23k-0069)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("lab7/assets", exist_ok=True)
os.makedirs("lab7/outputs", exist_ok=True)

# -------------------------------------------------------------
# 1. Generate Slanted Book / Document Image for Perspective Correction
# -------------------------------------------------------------
def generate_slanted_book():
    print("[*] Generating slanted book image...")
    # Create fronto-parallel book page
    pw, ph = 600, 800
    page = np.full((ph, pw, 3), 248, dtype=np.uint8) # ivory paper

    # Header banner
    cv2.rectangle(page, (40, 40), (pw - 40, 160), (45, 80, 180), -1)
    cv2.putText(page, "COMPUTER VISION", (65, 95), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(page, "LAB 07: HOMOGRAPHY & 3D SFM", (65, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 240, 255), 2, cv2.LINE_AA)

    # Sub-heading
    cv2.putText(page, "PERSPECTIVE RECTIFICATION TEST", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(page, (50, 225), (pw - 50, 225), (100, 100, 100), 2)

    # Text paragraphs (simulated lines and text)
    lines = [
        "Planar homography relates two perspective projections of a planar surface.",
        "Given four non-collinear point correspondences, we can compute matrix H.",
        "This allows us to warp any tilted photograph back to fronto-parallel view.",
        "Key Invariants: Collinearity is preserved, concurrency is preserved,",
        "cross-ratios are preserved, but angles and parallelisms are altered.",
        "FAST NUCES Karachi - Department of Artificial Intelligence",
        "Author: Muhammad John Raza | Roll No: 23k-0069"
    ]
    y_pos = 265
    for line in lines:
        cv2.putText(page, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 60), 1, cv2.LINE_AA)
        y_pos += 36

    # Draw color palette and test patterns
    colors = [(220, 50, 50), (50, 180, 50), (50, 50, 220), (220, 180, 40), (180, 40, 200), (40, 200, 220)]
    for i, c in enumerate(colors):
        x = 55 + i * 82
        cv2.rectangle(page, (x, 520), (x + 65, 590), c, -1)
        cv2.rectangle(page, (x, 520), (x + 65, 590), (0, 0, 0), 1)

    # Diagram box with concentric shapes
    cv2.rectangle(page, (50, 620), (pw - 50, 750), (230, 230, 230), -1)
    cv2.rectangle(page, (50, 620), (pw - 50, 750), (120, 120, 120), 2)
    center = (pw // 2, 685)
    cv2.circle(page, center, 45, (160, 60, 40), 3)
    cv2.circle(page, center, 25, (40, 120, 200), -1)
    cv2.rectangle(page, (center[0] - 80, center[1] - 30), (center[0] + 80, center[1] + 30), (60, 140, 60), 2)
    cv2.putText(page, "PLANAR TARGET PATTERN", (pw//2 - 110, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    # Add page border
    cv2.rectangle(page, (2, 2), (pw - 3, ph - 3), (180, 180, 180), 3)

    # Now composite onto a realistic desktop background with slanted perspective
    bg_w, bg_h = 1000, 800
    # Wooden desk texture
    desk = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    for row in range(bg_h):
        shade = int(55 + 25 * np.sin(row / 45.0) + (row % 12) * 1.5)
        desk[row, :, :] = [shade - 15, shade, shade + 35]

    # Destination coordinates for the tilted quad in background
    src_pts = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst_pts = np.float32([[260, 90], [820, 160], [710, 720], [130, 580]])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_page = cv2.warpPerspective(page, H, (bg_w, bg_h), flags=cv2.INTER_LANCZOS4)

    # Mask for compositing
    mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_pts)], 255)
    # Add subtle shadow
    shadow_pts = dst_pts + np.float32([18, 18])
    shadow_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    cv2.fillPoly(shadow_mask, [np.int32(shadow_pts)], 180)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (35, 35), 0)
    for c in range(3):
        desk[:, :, c] = np.clip(desk[:, :, c].astype(np.float32) * (1.0 - shadow_mask/255.0 * 0.45), 0, 255).astype(np.uint8)

    # Blend warped page into desk
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
    composite = (warped_page * mask_3d + desk * (1.0 - mask_3d)).astype(np.uint8)

    # Add a coffee cup / pen nearby for realism
    cv2.circle(composite, (900, 480), 55, (220, 220, 230), -1)
    cv2.circle(composite, (900, 480), 45, (30, 40, 50), -1)
    cv2.circle(composite, (900, 480), 55, (140, 140, 150), 3)

    cv2.imwrite("lab7/assets/book_slanted.jpg", composite)
    cv2.imwrite("lab7/assets/book_original_flat.jpg", page)
    print("  -> Saved lab7/assets/book_slanted.jpg and flat reference.")

# -------------------------------------------------------------
# 2. Generate Checkerboard Calibration Images with Distortion
# -------------------------------------------------------------
def generate_checkerboard_calibration_dataset():
    print("[*] Generating checkerboard calibration dataset (9x6 corners)...")
    corners_x = 9
    corners_y = 6
    square_size = 50 # pixels per square on pattern

    # Pattern dimensions (10 x 7 squares to have 9 x 6 internal corners)
    pat_w = (corners_x + 1) * square_size
    pat_h = (corners_y + 1) * square_size
    board = np.full((pat_h, pat_w), 255, dtype=np.uint8)
    for r in range(corners_y + 1):
        for c in range(corners_x + 1):
            if (r + c) % 2 == 1:
                cv2.rectangle(board, (c * square_size, r * square_size),
                              ((c + 1) * square_size, (r + 1) * square_size), 0, -1)

    img_w, img_h = 800, 600

    # True camera intrinsics
    fx, fy = 750.0, 750.0
    cx, cy = img_w / 2.0, img_h / 2.0
    K_true = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # Known radial & tangential distortion
    k1, k2, p1, p2, k3 = -0.18, 0.04, 0.001, -0.001, 0.0

    # Generate 10 poses
    angles_and_translations = [
        (0, 0, 0, 0, 0, 850),
        (15, -10, 5, -40, -30, 820),
        (-12, 18, -4, 50, 20, 800),
        (20, 15, -12, -30, 40, 780),
        (-18, -14, 8, 40, -40, 830),
        (5, 22, -15, 60, -10, 810),
        (-8, -20, 10, -50, 30, 790),
        (12, 10, -8, 20, -50, 840),
        (-15, 12, 14, -40, 20, 820),
        (0, -16, -10, 30, 50, 790)
    ]

    for idx, (rx, ry, rz, tx, ty, tz) in enumerate(angles_and_translations):
        # Rotation matrix from Euler angles
        rvec = np.radians([rx, ry, rz])
        R, _ = cv2.Rodrigues(rvec)
        T = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)

        # Corners of board in 3D object space (centered)
        pts_3d = np.float32([
            [-pat_w/2, -pat_h/2, 0],
            [pat_w/2, -pat_h/2, 0],
            [pat_w/2, pat_h/2, 0],
            [-pat_w/2, pat_h/2, 0]
        ])

        # Project 4 corners to image using perspective
        pts_cam = (R @ pts_3d.T + T).T
        pts_proj = pts_cam[:, :2] / pts_cam[:, 2:3]
        u = fx * pts_proj[:, 0] + cx
        v = fy * pts_proj[:, 1] + cy
        dst_corners = np.c_[u, v].astype(np.float32)

        src_corners = np.float32([[0, 0], [pat_w, 0], [pat_w, pat_h], [0, pat_h]])
        H = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Warp pattern to image plane
        frame = cv2.warpPerspective(board, H, (img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=230)

        # Apply lens distortion via mesh grid
        grid_y, grid_x = np.indices((img_h, img_w), dtype=np.float32)
        x_norm = (grid_x - cx) / fx
        y_norm = (grid_y - cy) / fy
        r2 = x_norm**2 + y_norm**2
        radial = 1.0 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
        xd = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
        yd = y_norm * radial + p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
        map_x = xd * fx + cx
        map_y = yd * fy + cy

        distorted_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=220)
        # Add subtle sensor noise and blur
        noise = np.random.normal(0, 2, (img_h, img_w)).astype(np.float32)
        distorted_frame = np.clip(distorted_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        distorted_frame = cv2.cvtColor(distorted_frame, cv2.COLOR_GRAY2BGR)

        fn = f"lab7/assets/chessboard_{idx+1:02d}.jpg"
        cv2.imwrite(fn, distorted_frame)

    print(f"  -> Generated {len(angles_and_translations)} distorted checkerboard frames in lab7/assets/")

# -------------------------------------------------------------
# 3. Generate Multi-View Image Dataset for 3D SfM & COLMAP
# -------------------------------------------------------------
def generate_multiview_sfm_dataset():
    print("[*] Generating 3D object multi-view dataset (8 views)...")
    # We construct a 3D textured pavilion / architectural landmark with detailed features
    # so SIFT/ORB features match cleanly across consecutive views.
    w, h = 800, 600
    fx, fy = 700.0, 700.0
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Define 3D textured model: A multi-faceted structure with tower, pediment, pillars, walls
    # We define planar textured patches in 3D
    np.random.seed(42)
    num_views = 8
    radius = 3.8
    cam_height = 1.2

    # Pre-render textured building faces
    tex_w, tex_h = 300, 300
    textures = []
    for f in range(6):
        tex = np.full((tex_h, tex_w, 3), 200, dtype=np.uint8)
        # Add brick patterns
        for y in range(0, tex_h, 20):
            shift = 10 if (y // 20) % 2 == 1 else 0
            cv2.line(tex, (0, y), (tex_w, y), (140, 90, 80), 2)
            for x in range(shift, tex_w, 30):
                cv2.line(tex, (x, y), (x, y + 20), (140, 90, 80), 2)
        # Windows & emblems
        cv2.rectangle(tex, (60, 60), (120, 140), (40, 60, 100), -1)
        cv2.rectangle(tex, (180, 60), (240, 140), (40, 60, 100), -1)
        cv2.circle(tex, (150, 220), 40, (180, 50, 50), -1)
        cv2.putText(tex, f"FACE-{f+1}", (90, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        textures.append(tex)

    # 3D vertices of a central polygonal pavilion
    # 4 vertical walls and 4 roof triangles
    half_s = 1.0
    z_base = 0.0
    z_wall = 1.5
    z_roof = 2.4

    faces_3d = [
        # Front wall: (X, Y, Z)
        [np.array([[-half_s, -half_s, z_base], [half_s, -half_s, z_base], [half_s, -half_s, z_wall], [-half_s, -half_s, z_wall]]), textures[0]],
        # Right wall
        [np.array([[half_s, -half_s, z_base], [half_s, half_s, z_base], [half_s, half_s, z_wall], [half_s, -half_s, z_wall]]), textures[1]],
        # Back wall
        [np.array([[half_s, half_s, z_base], [-half_s, half_s, z_base], [-half_s, half_s, z_wall], [half_s, half_s, z_wall]]), textures[2]],
        # Left wall
        [np.array([[-half_s, half_s, z_base], [-half_s, -half_s, z_base], [-half_s, -half_s, z_wall], [-half_s, half_s, z_wall]]), textures[3]],
    ]

    angles = np.linspace(-60, 60, num_views)
    for idx, deg in enumerate(angles):
        rad = np.radians(deg)
        # Camera position on an arc
        cam_x = radius * np.sin(rad)
        cam_y = -radius * np.cos(rad)
        cam_z = cam_height

        C = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
        target = np.array([0, 0, 0.9], dtype=np.float64)
        up = np.array([0, 0, 1], dtype=np.float64)

        # Look-at matrix
        zaxis = (target - C)
        zaxis = zaxis / np.linalg.norm(zaxis)
        xaxis = np.cross(zaxis, up)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)

        R_cam = np.vstack([xaxis, -yaxis, zaxis])
        t_cam = -R_cam @ C

        # Background sky and ground
        view_img = np.zeros((h, w, 3), dtype=np.uint8)
        # Sky gradient
        for y in range(h):
            view_img[y, :] = [int(225 - 60 * y/h), int(210 - 40 * y/h), int(190 - 20 * y/h)]
        # Ground plane
        cv2.rectangle(view_img, (0, h//2), (w, h), (90, 115, 80), -1)

        # Sort faces back to front relative to camera
        def face_depth(item):
            pts = item[0]
            center = np.mean(pts, axis=0)
            return np.linalg.norm(center - C)

        sorted_faces = sorted(faces_3d, key=face_depth, reverse=True)

        for pts3d, tex in sorted_faces:
            # Check face normal direction for backface culling
            v1 = pts3d[1] - pts3d[0]
            v2 = pts3d[3] - pts3d[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            view_vec = C - np.mean(pts3d, axis=0)
            if np.dot(normal, view_vec) <= 0.05:
                continue # facing away

            # Project 3D points
            cam_pts = (R_cam @ pts3d.T + t_cam.reshape(3,1)).T
            if np.any(cam_pts[:, 2] <= 0.1):
                continue
            proj_x = fx * (cam_pts[:, 0] / cam_pts[:, 2]) + cx
            proj_y = fy * (cam_pts[:, 1] / cam_pts[:, 2]) + cy
            dst_pts = np.c_[proj_x, proj_y].astype(np.float32)

            src_pts = np.float32([[0, tex_h], [tex_w, tex_h], [tex_w, 0], [0, 0]])
            H_face = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(tex, H_face, (w, h))

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(dst_pts)], 255)
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0
            view_img = (warped * mask_3d + view_img * (1.0 - mask_3d)).astype(np.uint8)

        # Save view
        fn = f"lab7/assets/multiview_{idx+1:02d}.jpg"
        cv2.imwrite(fn, view_img)

    print(f"  -> Generated {num_views} multi-view SfM images in lab7/assets/")

if __name__ == "__main__":
    generate_slanted_book()
    generate_checkerboard_calibration_dataset()
    generate_multiview_sfm_dataset()
    print("[SUCCESS] All test assets created successfully.")
