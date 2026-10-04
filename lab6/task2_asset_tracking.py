"""
Task 2: Asset Tracking in a Computer Lab Using SIFT
==================================================
Technique: SIFT (Scale-Invariant Feature Transform) Feature Detection & Description +
           Brute-Force k-NN Matching + Lowe's Ratio Test + RANSAC Homography Estimation.

Problem Statement:
- Automatically identify and track individual IT assets (Monitors, Keyboards, CPU Towers)
  and their unique components in a lab environment.
- Maintain an accurate inventory status (PRESENT vs NOT FOUND) with spatial localization.

Parameters chosen and rationale:
- cv2.SIFT_create(): Computes 128-dimensional orientation-invariant, scale-invariant descriptors.
- BFMatcher(cv2.NORM_L2, crossCheck=False) with knnMatch(k=2):
  Finds the two closest nearest-neighbor descriptor candidates for each query keypoint.
- Lowe's Ratio Test (threshold = 0.75):
  Accepts a match only if distance(best) < 0.75 * distance(second_best). Eliminates ambiguous
  and repetitive false matches by requiring distinct superiority.
- cv2.findHomography(src, dst, cv2.RANSAC, 5.0):
  RANSAC fits a projective transform matrix H while rejecting spatial outliers. Reprojection
  threshold of 5.0 pixels tolerates minor perspective distortion and sensor noise.
- Confidence Threshold (min_inliers = 15):
  Requires at least 15 spatially consistent inliers to certify asset presence with zero false alarms.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_match_asset(ref_gray, scene_gray, ratio=0.75, ransac_thr=5.0):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(ref_gray, None)
    k2, d2 = sift.detectAndCompute(scene_gray, None)

    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, 0, [], (k1, k2)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(d1, d2, k=2)

    # Lowe's ratio test
    good_matches = [
        m for m, n in raw_matches
        if m.distance < ratio * n.distance
    ]

    if len(good_matches) < 4:
        return None, 0, good_matches, (k1, k2)

    src_pts = np.float32([k1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thr)
    inliers = int(mask.sum()) if mask is not None else 0

    return H, inliers, good_matches, (k1, k2)

def track_assets(scene_path="assets/lab_scene.jpg",
                 output_path="task2_asset_tracking_output.png",
                 min_inliers=15):
    scene_bgr = cv2.imread(scene_path)
    if scene_bgr is None:
        raise FileNotFoundError(f"Cannot load scene image {scene_path}")
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    catalogue = {
        "Monitor": "assets/ref_monitor.jpg",
        "Keyboard": "assets/ref_keyboard.jpg",
        "CPU Tower": "assets/ref_tower.jpg",
        "Webcam": "assets/ref_webcam.jpg"  # Demonstrates absent asset
    }

    colors = {
        "Monitor": (0, 220, 0),      # Bright Green
        "Keyboard": (255, 120, 0),   # Cyan/Orange
        "CPU Tower": (255, 0, 255),  # Magenta
        "Webcam": (0, 0, 255)        # Red
    }

    annotated = scene_bgr.copy()
    inventory_results = []

    print("\n" + "="*55)
    print("           LAB ASSET TRACKING INVENTORY REPORT")
    print("="*55)
    print(f"{'Asset Name':<15} | {'Inliers':<8} | {'Status':<12} | {'Action'}")
    print("-"*55)

    for name, ref_path in catalogue.items():
        ref_bgr = cv2.imread(ref_path)
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        rh, rw = ref_gray.shape[:2]

        H, inliers, good, (k1, k2) = sift_match_asset(ref_gray, scene_gray)
        is_present = (H is not None) and (inliers >= min_inliers)

        status_str = "PRESENT" if is_present else "NOT FOUND"
        action_str = "Logged in Station" if is_present else "Audit / Missing"
        print(f"{name:<15} | {inliers:<8} | {status_str:<12} | {action_str}")

        inventory_results.append({
            "name": name,
            "ref_img": ref_bgr,
            "inliers": inliers,
            "status": status_str,
            "present": is_present
        })

        if is_present:
            # Project reference rectangle corners onto scene
            box_pts = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
            transformed_pts = cv2.perspectiveTransform(box_pts, H)

            # Draw polygon and label
            cv2.polylines(annotated, [np.int32(transformed_pts)], True, colors[name], 3)
            # Find top-left of transformed box for label placement
            tx = int(np.min(transformed_pts[:, 0, 0]))
            ty = int(np.min(transformed_pts[:, 0, 1])) - 8
            label = f"{name} ({inliers} inliers)"
            cv2.putText(annotated, label, (max(10, tx), max(25, ty)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[name], 2)

    print("="*55 + "\n")

    # Plot figure: Top row = Catalogue assets, Bottom row = Lab Scene & Detections
    fig = plt.figure(figsize=(15, 10))

    # Catalogue previews
    for i, item in enumerate(inventory_results, 1):
        plt.subplot(2, 4, i)
        plt.imshow(cv2.cvtColor(item["ref_img"], cv2.COLOR_BGR2RGB))
        color_title = "green" if item["present"] else "red"
        plt.title(f"{item['name']}\n[{item['status']}]", fontsize=11, fontweight="bold", color=color_title)
        plt.axis("off")

    # Full scene results
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Lab Scene", fontsize=12, fontweight="bold")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Multi-Asset Tracking & Localization", fontsize=12, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    cv2.imwrite("task2_tracked_scene.png", annotated)
    print(f"Task 2 output saved to {output_path}")
    return inventory_results

if __name__ == "__main__":
    track_assets()
