"""
Task 1: Computer Screen Detection in a Computer Lab
==================================================
Technique: Canny Edge Detection + Hough Line Transformation (HoughLinesP) +
           Morphological Reconstruction + Contour Analysis + Intensity Profiling.

Problem Statement:
- Identify computer screens in rows of computer workstations.
- Classify screen status as 'ON' or 'OFF' based on display luminance.
- Detect anomalies such as missing screens compared to expected lab inventory.

Parameters chosen and rationale:
- Canny(50, 150): Accurately detects screen bezels and borders while filtering texture noise.
- HoughLinesP(threshold=25, minLineLength=25, maxLineGap=25):
  - threshold=25: Votes needed to identify line candidate.
  - minLineLength=25: Discards short edge fragments and text noise inside the room.
  - maxLineGap=25: Connects slightly broken border segments into continuous edges.
- Orientation Filter (angle < 15 deg or > 75 deg): Keeps only rectilinear screen borders.
- Morphological Close (kernel 9x9): Bridges corners between horizontal and vertical screen edges.
- Contour Filter (Area > 8,000, 1.1 < Aspect Ratio < 2.0): Eliminates desks, stands, and walls.
- Threshold mean luminance (90.0): Lit LCD screens show bright output (>140), whereas inactive
  screens remain dark (<70).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_screens(img_path="assets/lab_screens.jpg",
                   output_path="task1_screens_output.png",
                   expected_count=6):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image from {img_path}")

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Edge detection with pre-blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. Hough Line Detection (Probabilistic)
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=25, maxLineGap=25)
    line_mask = np.zeros_like(gray)

    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)
            # Filter out giant room-spanning desk lines (length > 250)
            if (angle < 15 or abs(angle - 180) < 15) and length > 250:
                continue
            # Keep only near-horizontal or near-vertical lines (monitor bezels)
            if angle < 15 or angle > 75 or abs(angle - 180) < 15:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

    # 3. Morphological closing to seal rectangular perimeters
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Extract screen contours and evaluate bounding boxes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img.copy()

    detected_screens = []
    screen_idx = 1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)

        # Standard monitor aspect ratio is 16:9 or 16:10 (~1.25 to 1.75)
        if area > 8000 and 1.1 < aspect_ratio < 2.0:
            roi = gray[y:y+h, x:x+w]
            mean_intensity = float(roi.mean())
            status = "ON" if mean_intensity > 90.0 else "OFF"
            color = (0, 220, 0) if status == "ON" else (0, 0, 240)  # Green for ON, Red for OFF

            # Draw bounding box and label
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            label = f"#{screen_idx} {status} ({mean_intensity:.0f})"
            cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            detected_screens.append({
                "id": screen_idx,
                "bbox": (x, y, w, h),
                "status": status,
                "mean_intensity": mean_intensity
            })
            screen_idx += 1

    found_count = len(detected_screens)
    missing_count = max(0, expected_count - found_count)

    # Summary banner on image
    banner_text = f"Total Detected: {found_count} | Expected: {expected_count} | Missing: {missing_count}"
    cv2.rectangle(annotated, (0, 0), (w_img, 45), (30, 30, 30), -1)
    cv2.putText(annotated, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Save visualization figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Lab Scene", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title("2. Canny Edge Map", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(closed, cmap="gray")
    axes[1, 0].set_title("3. Hough Lines + Morphological Closed Mask", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"4. Detected Screens (ON/OFF & Missing Count)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save direct annotated image
    cv2.imwrite("task1_screens_detected.png", annotated)

    print(f"Task 1 Complete:")
    print(f" - Found screens: {found_count} (Expected: {expected_count})")
    for s in detected_screens:
        print(f"   Screen {s['id']}: Status={s['status']}, Mean Luminance={s['mean_intensity']:.1f}, BBox={s['bbox']}")
    print(f" - Missing screens anomaly: {missing_count} screen(s) missing from lab inventory")
    print(f" - Saved output visualization: {output_path}")

    return detected_screens

if __name__ == "__main__":
    detect_screens()
