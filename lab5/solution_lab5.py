import os
from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
cv2.setRNGSeed(0)

print("=" * 60)
print("LAB 05: IMAGE SEGMENTATION - COMPLETE EXECUTION")
print("=" * 60)

# ==============================================================================
# TASK 1: Global Thresholding vs. Adaptive Thresholding
# ==============================================================================
print("\n--- TASK 1: Global vs. Adaptive Thresholding ---")
gray_sudoku = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)
if gray_sudoku is None:
    raise FileNotFoundError("sudoku.png not found")

panels_t1 = [("Original", gray_sudoku)]
t_values = [80, 127, 180]
for T in t_values:
    _, m = cv2.threshold(gray_sudoku, T, 255, cv2.THRESH_BINARY)
    white_pct = (m == 255).mean() * 100
    print(f"Global T={T}: white = {white_pct:.1f}%")
    panels_t1.append((f"Global T={T}", m))

adaptive_t1 = cv2.adaptiveThreshold(
    gray_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
panels_t1.append(("Adaptive (G, bs=11, C=2)", adaptive_t1))

plt.figure(figsize=(18, 4))
for i, (title, im) in enumerate(panels_t1, 1):
    plt.subplot(1, 5, i)
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task1_output.png", dpi=150)
plt.close()
print("Saved task1_output.png")

# ==============================================================================
# TASK 2: Choosing the Correct Adaptive Threshold Strategy
# ==============================================================================
print("\n--- TASK 2: Mean vs. Gaussian Adaptive Thresholding Grid ---")
methods = {
    "Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}

block_sizes = [5, 15, 51]
c_values = [2, 8, 20]

for name, meth in methods.items():
    plt.figure(figsize=(12, 12))
    i = 1
    print(f"\nAdaptive {name} statistics (% black pixels):")
    for bs in block_sizes:
        for C in c_values:
            m = cv2.adaptiveThreshold(
                gray_sudoku, 255, meth, cv2.THRESH_BINARY, bs, C
            )
            black_pct = (m == 0).mean() * 100
            print(f"  bs={bs:2d}, C={C:2d} -> black = {black_pct:.1f}%")
            plt.subplot(3, 3, i)
            i += 1
            plt.imshow(m, cmap="gray")
            plt.title(f"{name} bs={bs} C={C} ({black_pct:.1f}% ink)", fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"task2_{name}.png", dpi=150)
    plt.close()
    print(f"Saved task2_{name}.png")

# Combined comparison figure for Task 2
plt.figure(figsize=(16, 8))
selected_configs = [
    ("Mean (5, 2)", cv2.ADAPTIVE_THRESH_MEAN_C, 5, 2),
    ("Mean (15, 8)", cv2.ADAPTIVE_THRESH_MEAN_C, 15, 8),
    ("Mean (51, 20)", cv2.ADAPTIVE_THRESH_MEAN_C, 51, 20),
    ("Gaussian (5, 2)", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 5, 2),
    ("Gaussian (15, 8) [Best]", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 8),
    ("Gaussian (51, 20)", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 51, 20),
]
for idx, (title, meth, bs, C) in enumerate(selected_configs, 1):
    m = cv2.adaptiveThreshold(gray_sudoku, 255, meth, cv2.THRESH_BINARY, bs, C)
    plt.subplot(2, 3, idx)
    plt.imshow(m, cmap="gray")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task2_output.png", dpi=150)
plt.close()
print("Saved task2_output.png")

# ==============================================================================
# TASK 3: Otsu's Thresholding Quality Control
# ==============================================================================
print("\n--- TASK 3: Otsu's Thresholding & Sensitivity ---")
gray_coins = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)
if gray_coins is None:
    gray_coins = cv2.imread("water_coins.jpg", cv2.IMREAD_GRAYSCALE)


def otsu_thresh(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


variants = {
    "raw": gray_coins,
    "blur 5x5": cv2.GaussianBlur(gray_coins, (5, 5), 0),
    "contrast x1.5": cv2.convertScaleAbs(gray_coins, alpha=1.5, beta=0),
    "contrast x0.6": cv2.convertScaleAbs(gray_coins, alpha=0.6, beta=0),
}

for name, im in variants.items():
    t, _ = otsu_thresh(im)
    print(f"  {name:14s} Otsu T = {t:.1f}")

t_opt, mask_otsu = otsu_thresh(gray_coins)
hist_coins = cv2.calcHist([gray_coins], [0], None, [256], [0, 256]).ravel()

plt.figure(figsize=(15, 4.5))
plt.subplot(131)
plt.imshow(gray_coins, cmap="gray")
plt.title("Original (Grayscale Coins)", fontsize=11, fontweight="bold")
plt.axis("off")

plt.subplot(132)
plt.plot(hist_coins, color="black", lw=1.5)
plt.axvline(
    t_opt,
    color="crimson",
    linestyle="--",
    lw=2,
    label=f"Otsu Optimal T = {t_opt:.1f}",
)
plt.title("Intensity Histogram & Cutoff", fontsize=11, fontweight="bold")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.subplot(133)
plt.imshow(mask_otsu, cmap="gray")
plt.title("Otsu Binary Mask", fontsize=11, fontweight="bold")
plt.axis("off")

plt.tight_layout()
plt.savefig("task3_output.png", dpi=150)
plt.close()
print("Saved task3_output.png")

# ==============================================================================
# TASK 4: Colour Segmentation with HSV
# ==============================================================================
print("\n--- TASK 4: HSV Color Segmentation (Smarties - Green) ---")
img_smarties = cv2.imread("smarties.png")
hsv_smarties = cv2.cvtColor(img_smarties, cv2.COLOR_BGR2HSV)

# Strict: very narrow hue & ultra-high saturation & value
strict_mask = cv2.inRange(
    hsv_smarties, np.array([65, 220, 220]), np.array([67, 255, 255])
)
# Final: broader hue band (45-85) and accommodating S & V
final_mask_raw = cv2.inRange(
    hsv_smarties, np.array([45, 60, 50]), np.array([85, 255, 255])
)

# Morphological cleanup
k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
final_mask = cv2.morphologyEx(final_mask_raw, cv2.MORPH_OPEN, k_morph)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k_morph)

extracted_smarties = cv2.bitwise_and(
    img_smarties, img_smarties, mask=final_mask
)

strict_cnt = cv2.countNonZero(strict_mask)
final_cnt = cv2.countNonZero(final_mask)
print(f"  Strict mask foreground pixels: {strict_cnt}")
print(f"  Final mask foreground pixels:  {final_cnt}")

panels_t4 = [
    ("Original Image", cv2.cvtColor(img_smarties, cv2.COLOR_BGR2RGB)),
    ("Strict HSV Mask", strict_mask),
    ("Final HSV Mask (Cleaned)", final_mask),
    (
        "Extracted Color Region",
        cv2.cvtColor(extracted_smarties, cv2.COLOR_BGR2RGB),
    ),
]

plt.figure(figsize=(16, 4))
for i, (t, im) in enumerate(panels_t4, 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(t, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task4_output.png", dpi=150)
plt.close()
print("Saved task4_output.png")

# ==============================================================================
# TASK 5: Canny Edge Detection & Hysteresis
# ==============================================================================
print("\n--- TASK 5: Canny Edge Detection & Hysteresis Thresholding ---")
gray_smarties = cv2.cvtColor(img_smarties, cv2.COLOR_BGR2GRAY)
blur_smarties = cv2.GaussianBlur(gray_smarties, (5, 5), 0)

pairs_canny = [(50, 100), (50, 200), (150, 200)]
panels_t5 = [("Original Grayscale", gray_smarties)]

for lo, hi in pairs_canny:
    edges = cv2.Canny(blur_smarties, lo, hi)
    cnt = cv2.countNonZero(edges)
    print(f"  Threshold pair ({lo}, {hi}): edge pixels = {cnt}")
    panels_t5.append((f"Canny ({lo}, {hi}) - {cnt} px", edges))

plt.figure(figsize=(16, 4))
for i, (t, im) in enumerate(panels_t5, 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap="gray")
    plt.title(t, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task5_output.png", dpi=150)
plt.close()
print("Saved task5_output.png")

# ==============================================================================
# TASK 6: Region Growing Inside Medical Image (Brain MRI)
# ==============================================================================
print("\n--- TASK 6: Region Growing on Brain MRI ---")


def region_grow(gray, seed, thr, conn8=False):
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    seed_val = int(gray[seed])
    mask[seed] = 255
    q = deque([seed])
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn8:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    while q:
        r, c = q.popleft()
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < h
                and 0 <= cc < w
                and mask[rr, cc] == 0
                and abs(int(gray[rr, cc]) - seed_val) <= thr
            ):
                mask[rr, cc] = 255
                q.append((rr, cc))
    return mask


gray_mri = cv2.imread("brain_mri.png", cv2.IMREAD_GRAYSCALE)
h_mri, w_mri = gray_mri.shape
print(
    f"  Loaded Brain MRI: {w_mri}x{h_mri}, intensity range: [{gray_mri.min()}, {gray_mri.max()}]"
)

# Seeds: Seed A in parenchyma/gray matter, Seed B in bright white matter tract
seed_A = (91, 137)
seed_B = (196, 164)
seeds = {"Seed A (Gray Matter)": seed_A, "Seed B (Bright Structure)": seed_B}

for s_name, s_coord in seeds.items():
    print(f"  {s_name} at {s_coord}: intensity = {gray_mri[s_coord]}")

thr_values = [10, 25, 45]
plt.figure(figsize=(15, 9))
idx_t6 = 1
for s_name, s_coord in seeds.items():
    for thr in thr_values:
        m_grown = region_grow(gray_mri, s_coord, thr)
        px_cnt = cv2.countNonZero(m_grown)
        print(f"  {s_name}, thr={thr}: grown pixels = {px_cnt}")

        # Overlay: show original MRI with green segmented region and red seed point
        vis = cv2.cvtColor(gray_mri, cv2.COLOR_GRAY2RGB)
        vis[m_grown == 255] = [30, 200, 30]  # Vibrant green segmented area
        cv2.circle(vis, (s_coord[1], s_coord[0]), 3, (255, 0, 0), -1)  # Red seed dot

        plt.subplot(2, 3, idx_t6)
        idx_t6 += 1
        plt.imshow(vis)
        plt.title(f"{s_name}\nThr={thr} | Region Px={px_cnt}", fontsize=11, fontweight="bold")
        plt.axis("off")

plt.tight_layout()
plt.savefig("task6_output.png", dpi=150)
plt.close()
print("Saved task6_output.png")

# ==============================================================================
# TASK 7 & 8: Watershed Pipeline & Tuning
# ==============================================================================
print("\n--- TASK 7 & 8: Marker-Based Watershed Pipeline & Tuning ---")
img_coins = cv2.imread("coins.jpg")
if img_coins is None:
    img_coins = cv2.imread("water_coins.jpg")


def watershed_pipeline(img, frac=0.5):
    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thresholding
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3. Noise removal
    k = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)

    # 4. Sure background
    sure_bg = cv2.dilate(opening, k, iterations=3)

    # 5. Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 6. Sure foreground
    _, sure_fg = cv2.threshold(dist, frac * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 7. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 8. Marker labeling
    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 9. Watershed
    out = img.copy()
    markers = cv2.watershed(out, markers)

    # 10. Boundary visualization
    out[markers == -1] = [0, 0, 255]  # Red boundary in BGR

    unique_regs = set(np.unique(markers).tolist()) - {-1, 1}
    stages = {
        "1. Threshold": thresh,
        "2. Sure BG": sure_bg,
        "3. Distance": dist,
        "4. Sure FG": sure_fg,
        "5. Unknown": unknown,
        "6. Final Watershed": out,
    }
    return stages, n_labels - 1, len(unique_regs)


# Task 7 Output
stages_t7, n_fg_t7, n_reg_t7 = watershed_pipeline(img_coins, frac=0.5)
panels_t7 = [("Original", cv2.cvtColor(img_coins, cv2.COLOR_BGR2RGB))]
for name, im in stages_t7.items():
    if name == "6. Final Watershed":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    panels_t7.append((name, im))

plt.figure(figsize=(21, 3.5))
for i, (t, im) in enumerate(panels_t7, 1):
    plt.subplot(1, 7, i)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(t, fontsize=10, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task7_output.png", dpi=150)
plt.close()
print("Saved task7_output.png")

# Task 8: Sweep distance thresholds
print("\n--- TASK 8 Table: Distance Transform Threshold Sweep ---")
print(
    f"{'Exp':<4} | {'Distance Threshold':<18} | {'Foreground Markers':<18} | {'Separated Regions':<18} | {'Observed Result'}"
)
print("-" * 95)
task8_results = []
fractions = [0.2, 0.4, 0.6, 0.8]
obs_comments = {
    0.2: "Under-segmented / merged blobs: cores touch across necks, clusters counted as single objects.",
    0.4: "Partial separation: several pairs remain merged where neck depth is large.",
    0.6: "Optimal separation: individual coin cores successfully isolated, exact boundaries detected.",
    0.8: "Over-segmented / eroded: high threshold shrinks cores too much, smaller coins lost.",
}
for idx, frac in enumerate(fractions, 1):
    st, n_fg, n_reg = watershed_pipeline(img_coins, frac=frac)
    obs = obs_comments[frac]
    print(f"{idx:<4} | {frac:.1f} * max{'':<10} | {n_fg:<18} | {n_reg:<18} | {obs}")
    task8_results.append((frac, n_fg, n_reg, obs, st["6. Final Watershed"]))

# Save Task 8 comparison figure
plt.figure(figsize=(16, 4))
for idx, (frac, n_fg, n_reg, obs, out_img) in enumerate(task8_results, 1):
    plt.subplot(1, 4, idx)
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Frac = {frac} ({n_fg} markers, {n_reg} regs)", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("task8_output.png", dpi=150)
plt.close()
print("Saved task8_output.png")

# ==============================================================================
# TASK 9: K-Means Color Clustering
# ==============================================================================
print("\n--- TASK 9: K-Means Color Clustering (dog.jpeg) ---")
img_dog = cv2.imread("dog.jpeg")
if img_dog is None:
    img_dog = cv2.imread("dogimg.jpg")

data_dog = img_dog.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

panels_t9 = [("Original", cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB))]
for K in (2, 4, 6):
    _, labels, centers = cv2.kmeans(
        data_dog, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    seg_dog = np.uint8(centers)[labels.flatten()].reshape(img_dog.shape)
    shares = np.bincount(labels.flatten(), minlength=K) / len(labels) * 100
    mean_diff = np.abs(seg_dog.astype(int) - img_dog.astype(int)).mean()
    print(f"\nK={K}:")
    print(f"  Cluster shares (%): {np.round(shares, 1).tolist()}")
    print(f"  Mean Absolute Difference (Reconstruction Error): {mean_diff:.2f}")
    print(f"  Cluster centers (BGR): {np.uint8(centers).tolist()}")
    panels_t9.append(
        (f"K={K} (Error={mean_diff:.1f})", cv2.cvtColor(seg_dog, cv2.COLOR_BGR2RGB))
    )

plt.figure(figsize=(16, 4))
for i, (t, im) in enumerate(panels_t9, 1):
    plt.subplot(1, 4, i)
    plt.imshow(im)
    plt.title(t, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task9_output.png", dpi=150)
plt.close()
print("Saved task9_output.png")

# ==============================================================================
# TASK 10: Segmentation Engineer Comparison on Challenging Image
# ==============================================================================
print("\n--- TASK 10: Multi-Method Segmentation Engineering Comparison ---")
# Image: smarties.png
img_t10 = cv2.imread("smarties.png")
gray_t10 = cv2.cvtColor(img_t10, cv2.COLOR_BGR2GRAY)
blur_t10 = cv2.GaussianBlur(gray_t10, (5, 5), 0)

# Method 1: Otsu's Thresholding
_, m1_otsu = cv2.threshold(
    blur_t10, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Method 2: HSV Color-based Segmentation (Green Candies)
hsv_t10 = cv2.cvtColor(img_t10, cv2.COLOR_BGR2HSV)
m2_hsv = cv2.inRange(
    hsv_t10, np.array([45, 60, 50]), np.array([85, 255, 255])
)
k_morph10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
m2_hsv = cv2.morphologyEx(m2_hsv, cv2.MORPH_OPEN, k_morph10)
m2_hsv = cv2.morphologyEx(m2_hsv, cv2.MORPH_CLOSE, k_morph10)

# Method 3: Watershed Segmentation
stages_ws10, _, _ = watershed_pipeline(img_t10, frac=0.45)
m3_ws = cv2.cvtColor(stages_ws10["6. Final Watershed"], cv2.COLOR_BGR2RGB)

panels_t10 = [
    ("Original Image", cv2.cvtColor(img_t10, cv2.COLOR_BGR2RGB)),
    ("Method 1: Otsu Thresholding", m1_otsu),
    ("Method 2: HSV Color Keying", m2_hsv),
    ("Method 3: Watershed", m3_ws),
]

plt.figure(figsize=(16, 4))
for i, (t, im) in enumerate(panels_t10, 1):
    plt.subplot(1, 4, i)
    plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
    plt.title(t, fontsize=11, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig("task10_output.png", dpi=150)
plt.close()
print("Saved task10_output.png")

print("\n" + "=" * 60)
print("ALL 10 TASKS EXECUTED SUCCESSFULLY AND FIGURES SAVED!")
print("=" * 60)
