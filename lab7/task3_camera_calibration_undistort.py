"""
Task 3: Undistort Live Camera Feed & Camera Calibration
=======================================================
Author: Muhammad John Raza
Roll No: 23k-0069
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 07)
"""

import os
import glob
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
assets_dir = "assets" if os.path.basename(os.getcwd()) == "lab7" else "lab7/assets"
os.makedirs(output_dir, exist_ok=True)

def calibrate_camera(image_dir=None,
                     pattern_size=(9, 6),
                     square_size=25.0):
    if image_dir is None:
        image_dir = assets_dir

    print("\n[*] Starting Camera Calibration Pipeline...")
    print(f"    - Pattern inner corners: {pattern_size[0]} x {pattern_size[1]}")
    print(f"    - Square size: {square_size} mm")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    pattern_glob = os.path.join(image_dir, "chessboard_*.jpg")
    images = sorted(glob.glob(pattern_glob))
    if not images:
        pattern_glob = os.path.join(get_path("lab7/assets"), "chessboard_*.jpg")
        images = sorted(glob.glob(pattern_glob))

    if not images:
        raise FileNotFoundError(f"No chessboard images found with pattern {pattern_glob}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_shape = None
    successful_frames = 0
    sample_drawn_corners = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_frames += 1

            if sample_drawn_corners is None:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners2, ret)
                sample_drawn_corners = vis

    print(f"    - Successfully detected corners in {successful_frames} / {len(images)} images.")

    if successful_frames < 3:
        raise RuntimeError("Insufficient valid calibration frames detected.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print(f"[*] Calibration Complete:")
    print(f"    - Overall RMS re-projection error: {ret:.4f} pixels")
    print(f"    - Mean re-projection error: {mean_error:.4f} pixels")
    print(f"    - Focal Length fx: {mtx[0, 0]:.2f}, fy: {mtx[1, 1]:.2f}")
    print(f"    - Principal Point cx: {mtx[0, 2]:.2f}, cy: {mtx[1, 2]:.2f}")
    print(f"    - Distortion Coeffs (k1, k2, p1, p2, k3):\n      {dist.ravel()}")

    output_npz = os.path.join(output_dir, "camera_calibration.npz")
    output_json = os.path.join(output_dir, "camera_matrix.json")
    np.savez(output_npz, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, error=mean_error)
    print(f"    - Saved calibration parameters to {output_npz}")

    calib_dict = {
        "camera_matrix_K": mtx.tolist(),
        "distortion_coefficients_D": dist.ravel().tolist(),
        "focal_length": {"fx": float(mtx[0, 0]), "fy": float(mtx[1, 1])},
        "principal_point": {"cx": float(mtx[0, 2]), "cy": float(mtx[1, 2])},
        "rms_reprojection_error": float(ret),
        "mean_reprojection_error_pixels": float(mean_error),
        "image_resolution": {"width": img_shape[0], "height": img_shape[1]},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(output_json, "w") as jf:
        json.dump(calib_dict, jf, indent=4)
    print(f"    - Saved calibration JSON to {output_json}")

    return mtx, dist, img_shape, sample_drawn_corners

def create_synthetic_distorted_stream(output_video_path=None, width=800, height=600, num_frames=60):
    if output_video_path is None:
        output_video_path = os.path.join(output_dir, "distorted_synthetic_stream.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    for f in range(num_frames):
        frame = np.full((height, width, 3), 240, dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)

        cv2.rectangle(frame, (100, 100), (700, 500), (70, 70, 70), 3)
        cv2.putText(frame, "SIMULATED LIVE CAMERA FEED", (180, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (40, 40, 40), 2)

        bx = int(width / 2 + 200 * np.sin(f / 8.0))
        by = int(height / 2 + 120 * np.cos(f / 8.0))
        cv2.circle(frame, (bx, by), 35, (0, 120, 240), -1)
        cv2.putText(frame, f"Frame: {f+1}", (120, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        cx, cy = width / 2.0, height / 2.0
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        x_norm = (grid_x - cx) / 750.0
        y_norm = (grid_y - cy) / 750.0
        r2 = x_norm**2 + y_norm**2
        radial = 1.0 - 0.20 * r2 + 0.05 * (r2**2)
        map_x = (x_norm * radial) * 750.0 + cx
        map_y = (y_norm * radial) * 750.0 + cy
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderValue=(30, 30, 30))

        out.write(distorted)

    out.release()
    return output_video_path

def run_live_undistortion(camera_index=0, max_runtime_sec=4.0):
    print("\n[*] Initializing Live Camera Feed Undistortion via cv2.VideoCapture()...")

    calib_file = os.path.join(output_dir, "camera_calibration.npz")
    if not os.path.exists(calib_file):
        mtx, dist, img_shape, sample_corners = calibrate_camera()
    else:
        data = np.load(calib_file)
        mtx, dist = data['mtx'], data['dist']
        img_shape = (800, 600)
        sample_corners = None

    cap = cv2.VideoCapture(camera_index)
    using_synthetic = False

    if not cap.isOpened():
        print(f"    [!] Physical camera {camera_index} not accessible.")
        print("    [*] Automatically initiating simulated live video stream fallback...")
        stream_path = create_synthetic_distorted_stream()
        cap = cv2.VideoCapture(stream_path)
        using_synthetic = True
    else:
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("    [!] Physical camera failed to yield frame. Using simulated live stream.")
            cap.release()
            stream_path = create_synthetic_distorted_stream()
            cap = cv2.VideoCapture(stream_path)
            using_synthetic = True

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        w, h = 800, 600

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    out_video_path = os.path.join(output_dir, "q3_undistorted_feed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, 20.0, (w * 2, h))

    start_time = time.time()
    frame_count = 0
    saved_snapshot = None

    print(f"    [*] Video feed active. Applying real-time cv2.undistort() (Timeout: {max_runtime_sec}s)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_frame_start = time.time()

        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        fps = 1.0 / max(time.time() - t_frame_start, 1e-4)

        vis_distorted = frame.copy()
        cv2.putText(vis_distorted, "INPUT FEED (DISTORTED)", (30, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_distorted, "Notice curved straight lines", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

        vis_undistorted = undistorted.copy()
        cv2.putText(vis_undistorted, f"UNDISTORTED (cv2.undistort)", (30, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(vis_undistorted, f"FPS: {fps:.1f} | Frame #{frame_count}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2, cv2.LINE_AA)

        side_by_side = np.hstack([vis_distorted, vis_undistorted])
        out_writer.write(side_by_side)

        if saved_snapshot is None and frame_count >= 15:
            saved_snapshot = side_by_side.copy()

        if time.time() - start_time >= max_runtime_sec:
            break

    cap.release()
    out_writer.release()

    if saved_snapshot is None:
        saved_snapshot = side_by_side

    snapshot_path = os.path.join(output_dir, "q3_side_by_side_snapshot.jpg")
    cv2.imwrite(snapshot_path, saved_snapshot)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    calib_sample_path = get_path("lab7/assets/chessboard_01.jpg")
    test_calib_img = cv2.imread(calib_sample_path)
    undist_calib = cv2.undistort(test_calib_img, mtx, dist, None, newcameramtx)

    axes[0].imshow(cv2.cvtColor(test_calib_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Distorted Calibration Image\n(Barrel Distortion visible along board edges)",
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, color='red', linestyle='--', alpha=0.6)

    axes[1].imshow(cv2.cvtColor(undist_calib, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Undistorted Frame via cv2.undistort()\n(Lines are rectified and straight)",
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, color='green', linestyle='--', alpha=0.6)

    plt.suptitle("Q3: Camera Calibration & Live Feed Undistortion\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_fig = os.path.join(output_dir, "q3_camera_calibration_undistort.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[*] Processed {frame_count} video frames.")
    print(f"    - Saved side-by-side video: {out_video_path}")
    print(f"    - Saved comparison figure: {output_fig}")
    print("[SUCCESS] Task 3 completed successfully.")
    return output_fig

def main():
    print("=================================================================")
    print("      LAB 07 - TASK 3: CAMERA CALIBRATION & LIVE FEED           ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")
    calibrate_camera()
    run_live_undistortion(camera_index=0, max_runtime_sec=4.0)

if __name__ == "__main__":
    main()
