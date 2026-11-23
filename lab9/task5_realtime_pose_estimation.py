"""
Lab 09: Modern Object Detection & Real-Time Pose Estimation
Task 5: Real-Time Human Pose Estimation & Joint Angle Calculation from Webcam
Author: Muhammad John Raza (Roll No: 23k-0069)
FAST NUCES Karachi, Department of Artificial Intelligence
Course: Computer Vision (AI-4002) | Instructor: Talha Shahid

Requirements:
1. Access webcam using OpenCV (with simulated lab feed fallback).
2. Apply pose estimation frame-by-frame.
3. Draw skeletons and update in real-time.
4. Calculate and display angles between joints (e.g., Left & Right Elbow angle).
5. Output live video feed with human skeleton overlay and real-time coordinates update.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("lab9") / rel_path
    if p2.exists():
        return p2
    if Path("lab9").is_dir():
        return Path("lab9") / rel_path
    return p1

# Standard COCO 17 Keypoints Definition
KEYPOINT_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# Anatomical skeleton connections (Pairs of keypoint indices)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial keypoints
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9),                            # Left arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                           # Right arm (Shoulder -> Elbow -> Wrist)
    (11, 13), (13, 15),                        # Left leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16)                         # Right leg (Hip -> Knee -> Ankle)
]

def calculate_angle(p1, p2, p3):
    """
    Computes angle at joint p2 formed by vectors (p1 - p2) and (p3 - p2).
    p1, p2, p3: (x, y) coordinates. Returns angle in degrees [0, 180].
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))

def open_webcam_stream():
    cap = cv2.VideoCapture(0)
    is_live = False
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            is_live = True
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("[INFO] Physical webcam accessed successfully (Device 0).")
            return cap, True
        cap.release()

    fallback = get_path("assets/pose_webcam_stream.mp4")
    if not fallback.exists():
        fallback = get_path("assets/pose_person.jpg")
    print(f"[INFO] Hardware webcam not active. Falling back to video feed: {fallback}")
    cap = cv2.VideoCapture(str(fallback))
    return cap, False

def run_realtime_pose_estimation(max_frames=60, show_window=False):
    model_path = get_path("models/yolov8n-pose.pt")
    if not model_path.exists():
        model_path = "yolov8n-pose.pt"

    print(f"[INFO] Loading Pose Estimation model from {model_path}...")
    model = YOLO(str(model_path))

    cap, is_live = open_webcam_stream()
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0

    output_dir = get_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = output_dir / "realtime_pose_skeleton_feed.mp4"
    snapshot_out_path = output_dir / "pose_estimation_skeleton_snapshot.png"
    csv_out_path = output_dir / "pose_coordinates_stream.csv"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps_in, (width, height))

    frame_count = 0
    prev_time = time.time()
    coordinates_records = []
    best_snapshot = None
    max_kpts_found = -1

    print("[INFO] Streaming Pose Estimation & Real-Time Joint Angle Analysis...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            if not is_live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        curr_time = time.time()
        fps_current = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        # Run inference
        t0 = time.time()
        results = model.predict(source=frame, conf=0.30, verbose=False)[0]
        dt_ms = (time.time() - t0) * 1000.0

        annotated = frame.copy()
        current_kpts_count = 0
        left_elbow_angle = None
        right_elbow_angle = None

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # Person keypoints data
            kpts_all = results.keypoints.data[0].cpu().numpy() # [17, 3] (x, y, conf)
            current_kpts_count = len(kpts_all)

            # Draw Skeleton Bones (Lines)
            for (idx1, idx2) in SKELETON_EDGES:
                x1, y1, c1 = kpts_all[idx1]
                x2, y2, c2 = kpts_all[idx2]
                if c1 > 0.35 and c2 > 0.35:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    # Limbs colored dynamically (Arms: Cyan, Legs: Yellow, Torso: Magenta)
                    if idx1 in (5, 7, 6, 8):
                        bone_color = (255, 255, 0) # Cyan
                    elif idx1 in (11, 13, 12, 14):
                        bone_color = (0, 255, 255) # Yellow
                    else:
                        bone_color = (255, 0, 255) # Magenta
                    cv2.line(annotated, p1, p2, bone_color, 3, cv2.LINE_AA)

            # Draw Joint Keypoints (Circles)
            for idx, (x, y, c) in enumerate(kpts_all):
                if c > 0.35:
                    cv2.circle(annotated, (int(x), int(y)), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (int(x), int(y)), 8, (255, 255, 255), 2, cv2.LINE_AA)

            # Step 5d: Calculate Elbow Joint Angles
            # Left Elbow Angle: Shoulder(5) -> Elbow(7) -> Wrist(9)
            s_l, e_l, w_l = kpts_all[5], kpts_all[7], kpts_all[9]
            if s_l[2] > 0.35 and e_l[2] > 0.35 and w_l[2] > 0.35:
                left_elbow_angle = calculate_angle((s_l[0], s_l[1]), (e_l[0], e_l[1]), (w_l[0], w_l[1]))
                cv2.putText(annotated, f"L-Elbow: {left_elbow_angle:.1f} deg",
                            (int(e_l[0]) - 80, int(e_l[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.circle(annotated, (int(e_l[0]), int(e_l[1])), 10, (0, 255, 0), 2)

            # Right Elbow Angle: Shoulder(6) -> Elbow(8) -> Wrist(10)
            s_r, e_r, w_r = kpts_all[6], kpts_all[8], kpts_all[10]
            if s_r[2] > 0.35 and e_r[2] > 0.35 and w_r[2] > 0.35:
                right_elbow_angle = calculate_angle((s_r[0], s_r[1]), (e_r[0], e_r[1]), (w_r[0], w_r[1]))
                cv2.putText(annotated, f"R-Elbow: {right_elbow_angle:.1f} deg",
                            (int(e_r[0]) + 15, int(e_r[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.circle(annotated, (int(e_r[0]), int(e_r[1])), 10, (0, 255, 0), 2)

            # Record Coordinate Stream for Analysis
            coordinates_records.append({
                'Frame': frame_count + 1,
                'FPS': round(fps_current, 1),
                'Latency_ms': round(dt_ms, 1),
                'Left_Elbow_Angle_deg': round(left_elbow_angle, 1) if left_elbow_angle is not None else np.nan,
                'Right_Elbow_Angle_deg': round(right_elbow_angle, 1) if right_elbow_angle is not None else np.nan,
                'Nose_X': int(kpts_all[0][0]), 'Nose_Y': int(kpts_all[0][1]),
                'Left_Wrist_X': int(w_l[0]), 'Left_Wrist_Y': int(w_l[1]),
                'Right_Wrist_X': int(w_r[0]), 'Right_Wrist_Y': int(w_r[1]),
            })

        # Heads-Up Display (HUD): Real-time coordinates update box
        cv2.rectangle(annotated, (15, 15), (460, 210), (20, 20, 25), -1)
        cv2.rectangle(annotated, (15, 15), (460, 210), (0, 255, 0), 2)

        cv2.putText(annotated, "FAST NUCES CV Lab - Real-Time Pose Tracker", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(annotated, f"Feed: {'Live Webcam' if is_live else 'Synthetic Feed'} | FPS: {fps_current:.1f} ({dt_ms:.1f} ms)",
                    (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        l_ang_str = f"{left_elbow_angle:.1f} deg" if left_elbow_angle is not None else "N/A"
        r_ang_str = f"{right_elbow_angle:.1f} deg" if right_elbow_angle is not None else "N/A"
        cv2.putText(annotated, f"Joint Angle (Left Elbow):  {l_ang_str}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Joint Angle (Right Elbow): {r_ang_str}", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if coordinates_records:
            last = coordinates_records[-1]
            cv2.putText(annotated, f"Live Coords: Nose=({last['Nose_X']}, {last['Nose_Y']})", (25, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
            cv2.putText(annotated, f"Left Wrist=({last['Left_Wrist_X']}, {last['Left_Wrist_Y']}) | Right=({last['Right_Wrist_X']}, {last['Right_Wrist_Y']})",
                        (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 255), 1)
            cv2.putText(annotated, "Tracking Status: Active (17 Keypoint Skeleton)", (25, 198),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 205, 50), 1)

        out_writer.write(annotated)

        if current_kpts_count > max_kpts_found or best_snapshot is None:
            max_kpts_found = current_kpts_count
            best_snapshot = annotated.copy()

        frame_count += 1
        if show_window:
            cv2.imshow("Real-Time Pose Estimation", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_writer.release()
    if show_window:
        cv2.destroyAllWindows()

    if best_snapshot is not None:
        cv2.imwrite(str(snapshot_out_path), best_snapshot)

    if coordinates_records:
        df_coords = pd.DataFrame(coordinates_records)
        df_coords.to_csv(str(csv_out_path), index=False)

    print("\n--- Task 5 Pose Estimation Summary ---")
    print(f"Total Frames Processed: {frame_count}")
    print(f"[OK] Saved annotated live pose video: {video_out_path}")
    print(f"[OK] Saved pose skeleton snapshot: {snapshot_out_path}")
    print(f"[OK] Saved real-time joint coordinates stream: {csv_out_path}")

if __name__ == "__main__":
    run_realtime_pose_estimation(max_frames=60, show_window=False)
