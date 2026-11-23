"""
Lab 09: Modern Object Detection & Real-Time Pose Estimation
Task 1: Live Webcam Lab Asset Detection using YOLO
Author: Muhammad John Raza (Roll No: 23k-0069)
Department of Artificial Intelligence, FAST NUCES Karachi
Course: Computer Vision (AI-4002) | Instructor: Talha Shahid

Requirements:
1. Use YOLO (any version) using live web camera to detect lab assets.
2. Label lab assets (laptop, keyboard, mouse, monitor/tv, chair, book, bottle, cell phone, person).
3. Display bounding boxes, class labels, and confidence scores.
4. Measure and display live FPS and inference latency.
5. Provide automatic fallback to simulated video feed if physical camera is offline.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Resolves correct relative path whether executed from repo root or inside lab9/
def get_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("lab9") / rel_path
    if p2.exists():
        return p2
    # If file doesn't exist yet, return relative to repo root if lab9 folder exists
    if Path("lab9").is_dir():
        return Path("lab9") / rel_path
    return p1

LAB_ASSET_CLASSES = {
    'laptop': (0, 215, 255),      # Gold
    'tv': (255, 144, 30),         # Dodger Blue (Monitors)
    'keyboard': (50, 205, 50),    # Lime Green
    'mouse': (147, 20, 255),      # Pink/Purple
    'chair': (220, 100, 50),      # Cyan/Blue
    'bottle': (255, 191, 0),      # Deep Sky Blue
    'cup': (238, 130, 238),       # Violet
    'book': (0, 165, 255),        # Orange
    'cell phone': (0, 69, 255),   # Orange-Red
    'person': (255, 255, 0),      # Cyan (Lab user / student)
}

def open_camera_stream():
    """
    Attempts to access the live webcam (index 0).
    Falls back gracefully to recorded lab webcam feed if camera is unavailable.
    """
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

    fallback_video = get_path("assets/lab_webcam_stream.mp4")
    if not fallback_video.exists():
        fallback_video = get_path("assets/lab_assets_01.jpg")
    
    print(f"[INFO] Hardware webcam not found or busy. Falling back to simulated lab stream: {fallback_video}")
    cap = cv2.VideoCapture(str(fallback_video))
    return cap, False

def run_yolo_lab_detection(max_frames=60, show_window=False):
    model_path = get_path("models/yolov8n.pt")
    if not model_path.exists():
        model_path = "yolov8n.pt"

    print(f"[INFO] Loading YOLO model from {model_path}...")
    model = YOLO(str(model_path))

    cap, is_live = open_camera_stream()
    if not cap.isOpened():
        print("[ERROR] Failed to open any video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0

    output_dir = get_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = output_dir / "yolo_webcam_lab_assets.mp4"
    snapshot_out_path = output_dir / "yolo_webcam_detection_snapshot.png"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps_in, (width, height))

    frame_count = 0
    total_inference_time = 0.0
    detected_asset_counts = {}
    best_snapshot = None
    max_detected = -1

    prev_time = time.time()
    print("[INFO] Starting real-time YOLO lab asset detection stream...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            # If simulated video ended, rewind or stop
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

        # Inference
        t0 = time.time()
        results = model.predict(source=frame, conf=0.35, verbose=False)[0]
        infer_latency = (time.time() - t0) * 1000.0
        total_inference_time += infer_latency

        annotated = frame.copy()
        current_assets = []

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                is_lab_asset = label in LAB_ASSET_CLASSES
                color = LAB_ASSET_CLASSES.get(label, (0, 255, 0))

                tag = f"[ASSET] {label.upper()}: {conf*100:.1f}%" if is_lab_asset else f"{label}: {conf*100:.1f}%"
                if is_lab_asset:
                    current_assets.append(label)
                    detected_asset_counts[label] = detected_asset_counts.get(label, 0) + 1

                # Draw bounding box
                thickness = 3 if is_lab_asset else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Draw label background box
                (w_txt, h_txt), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, max(y1 - 25, 0)), (x1 + w_txt + 8, max(y1, 25)), color, -1)
                cv2.putText(annotated, tag, (x1 + 4, max(y1 - 7, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Overlay Heads-Up Display (HUD) Dashboard
        cv2.rectangle(annotated, (15, 15), (420, 160), (20, 20, 25), -1)
        cv2.rectangle(annotated, (15, 15), (420, 160), (0, 255, 255), 2)

        cv2.putText(annotated, "FAST NUCES CV Lab - YOLO Asset Detector", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        feed_mode = "Physical Webcam (Device 0)" if is_live else "Simulated Lab Stream"
        cv2.putText(annotated, f"Source: {feed_mode}", (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(annotated, f"FPS: {fps_current:.1f} | Latency: {infer_latency:.1f} ms", (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detected Lab Assets: {len(current_assets)} items", (25, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        summary_txt = ", ".join(set(current_assets)) if current_assets else "Scanning..."
        cv2.putText(annotated, f"Active: {summary_txt[:35]}", (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 215, 0), 1)

        out_writer.write(annotated)

        if len(current_assets) > max_detected or best_snapshot is None:
            max_detected = len(current_assets)
            best_snapshot = annotated.copy()

        frame_count += 1
        if show_window:
            cv2.imshow("YOLO Live Lab Asset Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_writer.release()
    if show_window:
        cv2.destroyAllWindows()

    if best_snapshot is not None:
        cv2.imwrite(str(snapshot_out_path), best_snapshot)

    avg_latency = total_inference_time / max(frame_count, 1)
    avg_fps = 1000.0 / max(avg_latency, 1e-5)

    print("\n--- Task 1 Detection Summary ---")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Average Inference Latency: {avg_latency:.2f} ms per frame")
    print(f"Average Processing Speed: {avg_fps:.1f} FPS")
    print("Detected Lab Assets Distribution across frames:")
    for asset, count in detected_asset_counts.items():
        print(f"  - {asset}: {count} detections")
    print(f"[OK] Saved detection video: {video_out_path}")
    print(f"[OK] Saved detection snapshot: {snapshot_out_path}")

if __name__ == "__main__":
    run_yolo_lab_detection(max_frames=60, show_window=False)
