"""
Foundation Task: Boundary Detection using Sobel, Canny, and LoG
==============================================================
Parameters chosen and rationale:
- cv2.CV_64F: Gradients take negative values when transitioning from bright to dark;
  uint8 would clip negatives to 0, losing half the edge information.
- GaussianBlur kernel (5, 5), sigma=1.4: Suppresses sensor noise while preserving structural edges.
- Canny thresholds (50, 150): Follows the recommended 1:3 ratio. Strong edges (>150) are kept;
  weak edges (50-150) are kept only if connected to strong edges, eliminating noise specks.
- LoG: Second derivative operator highlights zero-crossings corresponding to inflection points.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_task0(input_path="assets/edges_input.jpg", output_path="task0_edges_output.png"):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {input_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sobel edge detection (first derivatives)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
    sx_display = cv2.convertScaleAbs(sx)
    sy_display = cv2.convertScaleAbs(sy)

    # 2. Canny edge detector (smoothing + gradient + NMS + hysteresis)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    canny = cv2.Canny(blur, 50, 150)

    # 3. Laplacian of Gaussian (LoG - second derivative)
    log = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))

    # Visualization: 2x3 grid
    show = [
        ("Original Image", rgb),
        ("Sobel X (Vertical Edges)", sx_display),
        ("Sobel Y (Horizontal Edges)", sy_display),
        ("Sobel Magnitude (Combined)", mag),
        ("Canny Edges (Clean & Thin)", canny),
        ("LoG (Laplacian of Gaussian)", log)
    ]

    plt.figure(figsize=(15, 9))
    for i, (title, im) in enumerate(show, 1):
        plt.subplot(2, 3, i)
        plt.imshow(im, cmap=None if im.ndim == 3 else "gray")
        plt.title(title, fontsize=12, fontweight="bold")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Task 0 completed successfully. Saved output to {output_path}")

if __name__ == "__main__":
    run_task0()
