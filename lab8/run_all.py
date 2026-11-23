"""
Master Runner Script for Lab 08: Object Detection using R-CNN
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
    ret = os.system(f"python lab8/{script_name}")
    elapsed = time.time() - t0
    if ret != 0:
        print(f"[ERROR] {script_name} failed with exit code {ret}")
        return False
    print(f"[COMPLETED] {script_name} in {elapsed:.2f}s")
    return True

def main():
    print("======================================================================")
    print("         FAST NUCES KARACHI - DEPARTMENT OF AI                       ")
    print("         COMPUTER VISION LAB 08 - OBJECT DETECTION SUITE             ")
    print("         Author: Muhammad John Raza (Roll No: 23k-0069)              ")
    print("======================================================================")

    tasks = [
        ("task0_data_exploration.py", "Task 0: Data Exploration & GT BBox Inspection"),
        ("task1_selective_search.py", "Task 1: Selective Search & IoU Matching"),
        ("task2_rcnn_training.py", "Task 2: R-CNN Multi-Task Training"),
        ("task3_rcnn_inference.py", "Task 3: R-CNN Inference & Non-Maximum Suppression (NMS)"),
        ("task4_architectural_comparison.py", "Task 4: Architectural Comparison (CNN vs R-CNN vs Fast vs Faster)")
    ]

    results = []
    for script, desc in tasks:
        success = run_task(script, desc)
        results.append((script, desc, success))

    print("\n" + "=" * 70)
    print("                     LAB 08 SUMMARY OF RESULTS                       ")
    print("=" * 70)
    for script, desc, success in results:
        status_str = "SUCCESS [PASS]" if success else "FAILED [FAIL]"
        print(f" - {script:<35} | {status_str} | {desc[:40]}")
    print("=" * 70)

if __name__ == "__main__":
    main()
