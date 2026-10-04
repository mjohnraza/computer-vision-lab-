"""
Task 6: Autonomous Vehicle Lane Detection Using Hough Line Transformation
========================================================================
Technique: Grayscale Conversion + Gaussian Blur + Canny Edge Detection +
           Trapezoidal Region-of-Interest (ROI) Mask + Probabilistic Hough Lines (HoughLinesP) +
           Slope Polarity Classification + Linear Regression Averaging & Extrapolation.

Problem Statement:
- Detect road lane markings (both continuous solid lines and intermittent dashed lines)
  from a vehicle's forward-facing dashboard camera.
- Accurately fit continuous lane boundaries to guide vehicle lateral control and lane-keeping assistance.

Parameters chosen and rationale:
- GaussianBlur(5, 5, sigma=0): Attenuates pavement grain and high-frequency asphalt noise.
- Canny(50, 150): Preserves high-contrast road-marking boundaries while suppressing subtle tarmac texture.
- Trapezoidal ROI Mask: Limits edge processing exclusively to the drivable surface in front of the vehicle,
  ignoring horizon, trees, clouds, oncoming traffic, and guardrails.
- HoughLinesP(threshold=25, minLineLength=25, maxLineGap=80):
  - threshold=25: Accumulator threshold to register collinear edge pixels.
  - minLineLength=25: Filters transient lane artifacts and small road debris.
  - maxLineGap=80: Bridges the physical gaps between dashed lane dashes into unified lane trajectories.
- Slope Polarity Classification (Left: m < 0, Right: m > 0):
  Due to the inverted y-axis convention (y increases downwards), the left lane heading towards
  the vanishing point has negative slope (dx > 0, dy < 0), while the right lane has positive slope.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lanes(img_path="assets/road.jpg",
                 output_path="task6_lane_output.png",
                 output_overlay_path="task6_lane_detected.png"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {img_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Noise reduction & Canny edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. Region of Interest (ROI) Mask (Trapezoid targeting ego-lane)
    roi_mask = np.zeros_like(edges)
    horizon_y = int(h * 0.58)
    poly = np.array([
        [(int(w * 0.05), h),
         (int(w * 0.95), h),
         (int(w * 0.58), horizon_y),
         (int(w * 0.42), horizon_y)]
    ], np.int32)
    cv2.fillPoly(roi_mask, poly, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # 3. Probabilistic Hough Line Transform
    segs = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=25, minLineLength=25, maxLineGap=80)

    # Visualization of raw Hough segments
    hough_vis = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    left_lines = []
    right_lines = []

    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if x1 == x2:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            # Filter near-horizontal noise
            if abs(slope) < 0.35 or abs(slope) > 2.5:
                continue
            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    # 4. Extrapolate lane boundaries from bottom of frame (y=h) to horizon (y=horizon_y)
    lane_overlay = np.zeros_like(img)
    final_annotated = img.copy()

    left_pts = None
    right_pts = None

    if left_lines:
        m_l, b_l = np.median(left_lines, axis=0)
        y1_l = h
        y2_l = int(horizon_y + 10)
        x1_l = int((y1_l - b_l) / m_l)
        x2_l = int((y2_l - b_l) / m_l)
        left_pts = ((x1_l, y1_l), (x2_l, y2_l))
        cv2.line(lane_overlay, (x1_l, y1_l), (x2_l, y2_l), (0, 0, 255), 8)

    if right_lines:
        m_r, b_r = np.median(right_lines, axis=0)
        y1_r = h
        y2_r = int(horizon_y + 10)
        x1_r = int((y1_r - b_r) / m_r)
        x2_r = int((y2_r - b_r) / m_r)
        right_pts = ((x1_r, y1_r), (x2_r, y2_r))
        cv2.line(lane_overlay, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 8)

    # 5. Fill green drivable corridor polygon between detected lanes
    if left_pts and right_pts:
        corridor = np.array([
            left_pts[0], left_pts[1], right_pts[1], right_pts[0]
        ], np.int32)
        cv2.fillPoly(lane_overlay, [corridor], (0, 180, 0))

    # Alpha blend overlay onto original image
    final_annotated = cv2.addWeighted(final_annotated, 1.0, lane_overlay, 0.45, 0)
    # Re-draw crisp line boundaries
    if left_pts:
        cv2.line(final_annotated, left_pts[0], left_pts[1], (0, 0, 255), 5)
    if right_pts:
        cv2.line(final_annotated, right_pts[0], right_pts[1], (0, 0, 255), 5)

    cv2.putText(final_annotated, "LANE KEEPING ASSIST: ACTIVE", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Save visualization grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Dashboard Camera Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Draw ROI outline on edges
    roi_edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.polylines(roi_edge_vis, [poly], True, (0, 255, 0), 2)
    axes[0, 1].imshow(cv2.cvtColor(roi_edge_vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Canny Edges + Trapezoidal ROI Mask", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(hough_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"3. HoughLinesP Segments ({len(segs)} lines)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(final_annotated, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. Fitted Lane Boundaries & Drivable Corridor", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    cv2.imwrite(output_overlay_path, final_annotated)

    print("Task 6 Complete:")
    print(f" - Hough segments detected: {len(segs) if segs is not None else 0}")
    print(f" - Left lane segments: {len(left_lines)} | Right lane segments: {len(right_lines)}")
    print(f" - Saved multi-panel visualization: {output_path}")
    print(f" - Saved annotated lane overlay: {output_overlay_path}")

    return {
        "segments": len(segs) if segs is not None else 0,
        "left_candidates": len(left_lines),
        "right_candidates": len(right_lines)
    }

if __name__ == "__main__":
    detect_lanes()
