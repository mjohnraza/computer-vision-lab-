"""
Task 7: Coin Detection and Automated Counting Using Hough Circle Transformation
==============================================================================
Technique: Median Filtering + Hough Gradient Circle Transform (cv2.HoughCircles) +
           Centroid Marking + Sequential Coin Indexing.

Problem Statement:
- Automatically identify, segment, and count circular coins in images across varying
  illumination, contrast, and spatial configurations.

Parameters chosen and rationale:
- cv2.medianBlur(gray, 5):
  Median filtering effectively removes surface specular glare, salt-and-pepper sensor noise,
  and coin surface engravings without blurring edge boundaries.
- cv2.HOUGH_GRADIENT:
  Reduces computational complexity from 3D accumulator (a, b, r) down to 2D accumulator
  by using edge gradient normal vectors to vote for center coordinates (a, b).
- dp = 1.2: Inverse ratio of accumulator resolution to image resolution. 1.2 provides
  slight spatial pooling to tolerate minor circular eccentricity.
- minDist = 28: Minimum allowable distance between detected coin centers. Prevents multiple
  concentric or duplicate circles from firing around the same physical coin.
- param1 = 50: Upper threshold of the internal Canny edge detector (lower threshold is 25).
- param2 = 30: Accumulator threshold for circle centers. Values too low produce false circles;
  values too high miss fainter coins.
- minRadius = 12, maxRadius = 38: Restricts search space strictly to expected coin diameter range.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_coins(img_path="coins.jpg",
                output_path="task7_coins_output.png",
                output_annotated_path="task7_coins_detected.png",
                min_dist=28,
                param1=50,
                param2=30,
                min_radius=12,
                max_radius=38):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {img_path}")

    # 1. Grayscale and Median Blur (preserves edges, removes glare)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # 2. Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    annotated = img.copy()
    coin_data = []
    total_coins = 0

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Sort coins top-to-bottom, left-to-right for consistent numbering
        circles = sorted(circles, key=lambda c: (c[1] // 50, c[0]))
        total_coins = len(circles)

        for i, (x, y, r) in enumerate(circles, 1):
            # Outer circle in green
            cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
            # Center point in red
            cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)
            # Number label
            cv2.putText(annotated, str(i), (x - 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            coin_data.append({"id": i, "x": int(x), "y": int(y), "radius": int(r)})

    # Summary banner on top
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(annotated, f"AUTOMATED COIN COUNT: {total_coins}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Coins Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("2. Median Filtered Grayscale", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Hough Circles ({total_coins} Coins Counted)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    cv2.imwrite(output_annotated_path, annotated)

    print("Task 7 Complete:")
    print(f" - Total coins detected and counted: {total_coins}")
    for c in coin_data[:5]:
        print(f"   Coin #{c['id']}: Center=({c['x']}, {c['y']}), Radius={c['radius']}px")
    if total_coins > 5:
        print(f"   ... and {total_coins - 5} additional coins verified.")
    print(f" - Saved visualization plot: {output_path}")
    print(f" - Saved annotated result image: {output_annotated_path}")

    return {
        "total_coins": total_coins,
        "coin_details": coin_data
    }

if __name__ == "__main__":
    count_coins()
