"""
Lab 09: Modern Object Detection & Real-Time Pose Estimation
Script: generate_assets.py
Author: Muhammad John Raza (Roll No: 23k-0069)
FAST NUCES Karachi, Department of Artificial Intelligence
Instructor: Talha Shahid

Purpose:
Prepares high-resolution real-world laboratory asset images and simulated
video feeds (for webcam fallback) featuring computer lab assets:
- Laptops, Desktop Monitors, Mechanical Keyboards, Wireless Mice, Chairs,
  Books, Water Bottles, Cell Phones, and Persons/Students.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
from pathlib import Path

def get_base_dir():
    # If run inside lab9/
    if Path("assets").exists() or Path("outputs").exists():
        return Path(".")
    # If run from repo root
    return Path("lab9")

def download_or_synthesize(url: str, save_path: Path, fallback_generator):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success = False
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        img = cv2.imread(str(save_path))
        if img is not None and img.shape[0] > 100:
            success = True
            print(f"[OK] Downloaded: {save_path.name} ({img.shape[1]}x{img.shape[0]})")
    except Exception as e:
        print(f"[Notice] Direct download failed for {save_path.name} ({e}), generating synthetic asset...")

    if not success:
        img = fallback_generator()
        cv2.imwrite(str(save_path), img)
        print(f"[OK] Generated synthetic asset: {save_path.name} ({img.shape[1]}x{img.shape[0]})")
    return cv2.imread(str(save_path))

def create_synthetic_lab_desk():
    h, w = 720, 1280
    canvas = np.full((h, w, 3), (240, 240, 245), dtype=np.uint8)
    # Floor and wall
    cv2.rectangle(canvas, (0, 0), (w, 350), (220, 225, 230), -1)
    # Desk surface
    cv2.rectangle(canvas, (100, 350), (1180, 700), (180, 160, 140), -1)
    cv2.rectangle(canvas, (100, 350), (1180, 700), (120, 100, 80), 3)

    # Monitor
    cv2.rectangle(canvas, (200, 120), (520, 360), (30, 30, 30), -1)
    cv2.rectangle(canvas, (215, 135), (505, 335), (200, 180, 120), -1)
    cv2.rectangle(canvas, (340, 360), (380, 420), (50, 50, 50), -1)
    cv2.rectangle(canvas, (300, 420), (420, 435), (40, 40, 40), -1)
    cv2.putText(canvas, "Code Terminal", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    # Laptop
    cv2.rectangle(canvas, (560, 280), (840, 470), (40, 40, 40), -1)
    cv2.rectangle(canvas, (575, 295), (825, 450), (210, 220, 240), -1)
    cv2.rectangle(canvas, (540, 470), (860, 510), (70, 70, 70), -1)
    cv2.putText(canvas, "CV Lab Workstation", (590, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

    # Keyboard
    cv2.rectangle(canvas, (320, 490), (520, 560), (50, 50, 55), -1)
    cv2.rectangle(canvas, (320, 490), (520, 560), (90, 90, 95), 2)
    cv2.putText(canvas, "Keyboard", (370, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Mouse
    cv2.ellipse(canvas, (560, 530), (25, 40), 15, 0, 360, (40, 40, 40), -1)
    cv2.putText(canvas, "Mouse", (540, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Bottle
    cv2.rectangle(canvas, (880, 370), (940, 520), (180, 100, 40), -1)
    cv2.rectangle(canvas, (900, 340), (920, 370), (200, 200, 200), -1)
    cv2.putText(canvas, "Bottle", (885, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Book / Notebook
    cv2.rectangle(canvas, (150, 510), (290, 640), (40, 80, 160), -1)
    cv2.putText(canvas, "CV Book", (170, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Chair in background
    cv2.rectangle(canvas, (960, 240), (1120, 500), (60, 60, 70), -1)
    cv2.rectangle(canvas, (980, 500), (1100, 660), (40, 40, 45), -1)
    cv2.putText(canvas, "Chair", (1010, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    return canvas

def create_synthetic_person_pose():
    h, w = 720, 1280
    canvas = np.full((h, w, 3), (235, 238, 242), dtype=np.uint8)
    # Background lab wall
    cv2.rectangle(canvas, (0, 0), (w, 550), (210, 215, 225), -1)
    cv2.line(canvas, (0, 550), (w, 550), (170, 175, 185), 3)

    # Head
    center_head = (640, 180)
    cv2.circle(canvas, center_head, 55, (190, 150, 120), -1)
    cv2.circle(canvas, center_head, 55, (140, 100, 80), 2)

    # Torso
    cv2.rectangle(canvas, (560, 240), (720, 450), (45, 120, 180), -1)

    # Left arm (raised/bent to show joint angles)
    # Shoulder: (560, 260), Elbow: (460, 320), Wrist: (480, 210)
    cv2.line(canvas, (560, 260), (460, 320), (45, 120, 180), 30)
    cv2.line(canvas, (460, 320), (480, 210), (190, 150, 120), 24)

    # Right arm (extended downwards/outwards)
    # Shoulder: (720, 260), Elbow: (800, 340), Wrist: (860, 430)
    cv2.line(canvas, (720, 260), (800, 340), (45, 120, 180), 30)
    cv2.line(canvas, (800, 340), (860, 430), (190, 150, 120), 24)

    # Legs
    cv2.line(canvas, (600, 450), (590, 600), (50, 50, 60), 36)
    cv2.line(canvas, (590, 600), (585, 710), (40, 40, 50), 30)

    cv2.line(canvas, (680, 450), (690, 600), (50, 50, 60), 36)
    cv2.line(canvas, (690, 600), (695, 710), (40, 40, 50), 30)

    # Lab equipment next to person: laptop on cart
    cv2.rectangle(canvas, (900, 380), (1150, 550), (140, 140, 145), -1)
    cv2.rectangle(canvas, (950, 300), (1100, 400), (30, 30, 30), -1)
    cv2.rectangle(canvas, (960, 310), (1090, 390), (220, 240, 255), -1)

    return canvas

def generate_video_stream(frame_generator, save_path: Path, num_frames=60, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 1280, 720
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

    base_frame = frame_generator()
    for i in range(num_frames):
        frame = base_frame.copy()
        # Add smooth subtle movement to simulate live camera drift / subject motion
        dx = int(12 * np.sin(i * 0.1))
        dy = int(6 * np.cos(i * 0.1))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Add timestamp & FPS watermark
        cv2.putText(shifted, f"Live Lab Camera Feed | Frame {i+1}/{num_frames} | {fps} FPS", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(shifted)
    out.release()
    print(f"[OK] Generated simulated webcam stream: {save_path.name} ({num_frames} frames @ {fps} FPS)")

def main():
    base = get_base_dir()
    assets_dir = base / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (base / "outputs").mkdir(parents=True, exist_ok=True)

    print("--- Preparing Lab 09 Real-World Computer Vision Assets ---")
    
    # 1. Lab Assets Scene 1: Laptop, keyboard, mouse, bottle, book
    url_desk1 = "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=1280&q=80"
    download_or_synthesize(url_desk1, assets_dir / "lab_assets_01.jpg", create_synthetic_lab_desk)

    # 2. Lab Assets Scene 2: Tech lab with student working, laptop, chair, monitor
    url_desk2 = "https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=1280&q=80"
    download_or_synthesize(url_desk2, assets_dir / "lab_assets_02.jpg", create_synthetic_lab_desk)

    # 3. Lab Assets Scene 3: Multi-object desktop setup (mug, phone, laptop, notebook)
    url_desk3 = "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=1280&q=80"
    download_or_synthesize(url_desk3, assets_dir / "lab_assets_03.jpg", create_synthetic_lab_desk)

    # 4. Pose Person: Standing person in room for pose estimation
    url_pose = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=1280&q=80"
    download_or_synthesize(url_pose, assets_dir / "pose_person.jpg", create_synthetic_person_pose)

    # 5. Simulated Webcam Video Streams (for environments without physical camera)
    generate_video_stream(create_synthetic_lab_desk, assets_dir / "lab_webcam_stream.mp4", num_frames=60, fps=20)
    generate_video_stream(create_synthetic_person_pose, assets_dir / "pose_webcam_stream.mp4", num_frames=60, fps=20)

    print(f"\n[SUCCESS] All assets successfully staged in {assets_dir.resolve()}")

if __name__ == "__main__":
    main()
