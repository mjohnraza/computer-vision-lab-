"""
FAST-NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision (AI-4002) | Instructor: Talha Shahid
Student: Muhammad John Raza (Roll No: 23k-0069)
Lab 09: Modern Object Detection (YOLO, SSD, RetinaNet) & Real-Time Pose Estimation
Master Execution Pipeline: run_all.py
"""

import sys
import subprocess
from pathlib import Path

TASKS = [
    ("Asset Staging & Fallback Stream Synthesis", "generate_assets.py"),
    ("Task 1: YOLO Live Lab Asset Detection", "task1_yolo_webcam_detection.py"),
    ("Task 2: Comparative Detection (YOLOv3 vs. SSD vs. RetinaNet)", "task2_multi_model_detection.py"),
    ("Task 3: Speed (FPS), Confidence, Model Size & Latency Benchmarks", "task3_benchmark_metrics.py"),
    ("Task 4: Analytical Discussion & Performance Evaluation", "task4_performance_analysis.py"),
    ("Task 5: Real-Time Human Pose Estimation & Joint Angles", "task5_realtime_pose_estimation.py"),
]

def main():
    print("=" * 80)
    print("FAST-NUCES KARACHI - DEPARTMENT OF ARTIFICIAL INTELLIGENCE")
    print("COMPUTER VISION (AI-4002) - LAB 09 MASTER PIPELINE")
    print("Student: Muhammad John Raza | Roll No: 23k-0069 | Instructor: Talha Shahid")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent

    for idx, (title, script_name) in enumerate(TASKS, 1):
        print(f"\n[{idx}/{len(TASKS)}] EXECUTING: {title} ({script_name})...")
        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)

        res = subprocess.run([sys.executable, str(script_path)], cwd=str(base_dir))
        if res.returncode != 0:
            print(f"[FAILED] {script_name} returned exit code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[COMPLETED] {title}")

    print("\n" + "=" * 80)
    print("[SUCCESS] All Lab 09 tasks executed successfully!")
    print(f"Outputs and visualizations saved to: {(base_dir / 'outputs').resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
