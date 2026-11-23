"""
Master Execution Script for Lab 07
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
"""

import os
import sys
import time

def run_task(script_name, description):
    print("\n" + "=" * 70)
    print(f"RUNNING: {description} ({script_name})")
    print("=" * 70)
    t0 = time.time()
    ret = os.system(f"python lab7/{script_name}")
    elapsed = time.time() - t0
    if ret != 0:
        print(f"[ERROR] {script_name} failed with exit code {ret}")
        return False
    print(f"[COMPLETED] {script_name} in {elapsed:.2f}s")
    return True

def main():
    print("======================================================================")
    print("         FAST NUCES KARACHI - DEPARTMENT OF AI                       ")
    print("         COMPUTER VISION LAB 07 - COMPLETE SUITE                     ")
    print("         Author: Muhammad John Raza (Roll No: 23k-0069)              ")
    print("======================================================================")

    tasks = [
        ("task1_image_transformations.py", "Q1: Image Transformations (2D, 3D, Homogeneous, Affine, Homography)"),
        ("task2_perspective_correction.py", "Q2: Perspective Correction (Unwarping) / Inverse Homography"),
        ("task3_camera_calibration_undistort.py", "Q3: Undistort Live Camera Feed & Calibration"),
        ("task4_colmap_3d_reconstruction.py", "Q4: Build 3D Model from Images (COLMAP / SfM)")
    ]

    results = []
    for script, desc in tasks:
        success = run_task(script, desc)
        results.append((script, desc, success))

    print("\n" + "=" * 70)
    print("                     LAB 07 SUMMARY OF RESULTS                       ")
    print("=" * 70)
    for script, desc, success in results:
        status_str = "SUCCESS [PASS]" if success else "FAILED [FAIL]"
        print(f" - {script:<35} | {status_str} | {desc[:40]}")
    print("=" * 70)

if __name__ == "__main__":
    main()
