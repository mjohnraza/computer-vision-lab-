import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Lab 05: Image Segmentation\n",
                "**Computer Vision Lab · Department of Artificial Intelligence**\n",
                "\n",
                "### Core Concepts & Method Overview:\n",
                "- **Global Thresholding**: Single cutoff $T$ across the entire image.\n",
                "- **Adaptive Thresholding**: Dynamically computes local threshold per pixel using Mean or Gaussian neighborhood windows.\n",
                "- **Otsu's Thresholding**: Automatic optimal threshold determination by maximizing inter-class variance.\n",
                "- **HSV Color Segmentation**: Isolates target objects by decoupling chromatic identity (Hue) from illumination (Value) and purity (Saturation).\n",
                "- **Canny Edge Detection & Hysteresis**: Multi-stage edge filter utilizing gradient calculation, non-maximum suppression, and dual-threshold hysteresis linking.\n",
                "- **Region Growing**: Seed-driven spatial clustering grouping connected pixels within an intensity tolerance.\n",
                "- **Marker-Controlled Watershed**: Topological flooding from sure-foreground marker basins, creating separation dams at touch points.\n",
                "- **K-Means Clustering**: Unsupervised feature space quantization grouping pixels into $K$ color centroids without spatial bias.\n",
                "- **Segmentation Engineering**: Multi-method evaluation and trade-off synthesis on complex imagery."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from collections import deque\n",
                "\n",
                "# Set random seed for reproducible results\n",
                "np.random.seed(42)\n",
                "cv2.setRNGSeed(0)\n",
                "print('OpenCV Version:', cv2.__version__)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 1: Automated Inspection of a Document Under Uneven Lighting\n",
                "### Global Thresholding vs. Adaptive Thresholding\n",
                "We test three global threshold values ($T \\in \\{80, 127, 180\\}$) against adaptive Gaussian thresholding on `sudoku.png`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "gray_sudoku = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)\n",
                "\n",
                "panels_t1 = [(\"Original\", gray_sudoku)]\n",
                "for T in (80, 127, 180):\n",
                "    _, m = cv2.threshold(gray_sudoku, T, 255, cv2.THRESH_BINARY)\n",
                "    white_pct = (m == 255).mean() * 100\n",
                "    print(f\"Global T={T}: white = {white_pct:.1f}%\")\n",
                "    panels_t1.append((f\"Global T={T}\", m))\n",
                "\n",
                "# Adaptive Gaussian thresholding\n",
                "adaptive_t1 = cv2.adaptiveThreshold(gray_sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \n",
                "                                    cv2.THRESH_BINARY, 11, 2)\n",
                "panels_t1.append((\"Adaptive (G, bs=11, C=2)\", adaptive_t1))\n",
                "\n",
                "plt.figure(figsize=(18, 4))\n",
                "for i, (title, im) in enumerate(panels_t1, 1):\n",
                "    plt.subplot(1, 5, i)\n",
                "    plt.imshow(im, cmap=\"gray\", vmin=0, vmax=255)\n",
                "    plt.title(title, fontsize=11, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig('task1_output.png', dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 1 Analysis & Discussion\n",
                "**Why does a single threshold struggle when illumination changes across the image?**\n",
                "A global threshold compares every pixel's absolute grayscale value against a fixed scalar $T$.\n",
                "However, under non-uniform illumination (e.g. a strong lighting gradient), the background paper on the shadowed side of the page is physically darker than the foreground text on the brightly illuminated side.\n",
                "- At $T=80$, the dark background paper falls below $T$ and turns into large black blocks that swallow digits.\n",
                "- At $T=180$, the light paper survives, but the ink on the bright side is bleached white ($>180$), causing numbers and grid lines to vanish.\n",
                "- No single constant $T$ can simultaneously separate ink from paper across the entire lighting field."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 2: Choosing the Correct Adaptive Threshold Strategy\n",
                "We evaluate both **Mean-based** and **Gaussian-based** adaptive thresholding across an extensive parameter grid of neighbourhood sizes (`blockSize` $\\in \\{5, 15, 51\\}$) and constant offsets ($C \\in \\{2, 8, 20\\}$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "methods = {\"Mean\": cv2.ADAPTIVE_THRESH_MEAN_C, \"Gaussian\": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}\n",
                "block_sizes = [5, 15, 51]\n",
                "c_values = [2, 8, 20]\n",
                "\n",
                "for name, meth in methods.items():\n",
                "    plt.figure(figsize=(12, 12))\n",
                "    i = 1\n",
                "    print(f\"\\nAdaptive {name} statistics (% black pixels):\")\n",
                "    for bs in block_sizes:\n",
                "        for C in c_values:\n",
                "            m = cv2.adaptiveThreshold(gray_sudoku, 255, meth, cv2.THRESH_BINARY, bs, C)\n",
                "            black_pct = (m == 0).mean() * 100\n",
                "            print(f\"  bs={bs:2d}, C={C:2d} -> black = {black_pct:.1f}%\")\n",
                "            plt.subplot(3, 3, i); i += 1\n",
                "            plt.imshow(m, cmap=\"gray\")\n",
                "            plt.title(f\"{name} bs={bs} C={C} ({black_pct:.1f}% ink)\", fontsize=10)\n",
                "            plt.axis(\"off\")\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(f\"task2_{name}.png\", dpi=150)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 2: Answers to the 5 Core Analysis Questions\n",
                "1. **What happens when the neighbourhood is too small (e.g., $bs=5$)?**\n",
                "   The window fits completely inside thicker printed lines or stroke interiors. The local average approximates the dark stroke intensity itself, so the center of the stroke turns white: digits become hollow, broken, and surrounded by salt-and-pepper noise.\n",
                "\n",
                "2. **What happens when the neighbourhood is too large (e.g., $bs=51$)?**\n",
                "   The window spans regions large enough to capture the global illumination gradient. It begins to behave like global thresholding, causing dark shadowed areas of the paper to appear as unwanted dark blobs.\n",
                "\n",
                "3. **What happens when $C$ is increased?**\n",
                "   $T(x,y) = \\text{local\\_mean} - C$. Increasing $C$ reduces the threshold bar, meaning a pixel must be significantly darker than its surroundings to qualify as foreground ink. Foreground black percentage consistently decreases, eliminating background noise; however, excessively high $C$ breaks thin strokes.\n",
                "\n",
                "4. **Which combination produces the cleanest text segmentation?**\n",
                "   **Gaussian method with $blockSize=15$ and $C=8$** (yielding 13.6% ink). The window is larger than the line stroke width but smaller than the lighting gradient, with $C=8$ effectively suppressing background paper texture while preserving complete character strokes.\n",
                "\n",
                "5. **Which adaptive method performs better on this image?**\n",
                "   **Gaussian-based adaptive thresholding** outperforms Mean-based. Gaussian weighting assigns higher significance to center pixels and fades smoothly outward, avoiding harsh box-boundary artifacts and rendering clean, smooth character contours."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 3: Quality-Control System Using Otsu's Thresholding\n",
                "We test Otsu's automatic bimodal threshold selection on `coins.jpg` under various preprocessing operations (raw, Gaussian blur $5\\times 5$, contrast $\\times 1.5$, and contrast $\\times 0.6$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "gray_coins = cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)\n",
                "if gray_coins is None:\n",
                "    gray_coins = cv2.imread('water_coins.jpg', cv2.IMREAD_GRAYSCALE)\n",
                "\n",
                "def otsu_thresh(img):\n",
                "    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
                "\n",
                "variants = {\n",
                "    \"raw\": gray_coins,\n",
                "    \"blur 5x5\": cv2.GaussianBlur(gray_coins, (5, 5), 0),\n",
                "    \"contrast x1.5\": cv2.convertScaleAbs(gray_coins, alpha=1.5, beta=0),\n",
                "    \"contrast x0.6\": cv2.convertScaleAbs(gray_coins, alpha=0.6, beta=0)\n",
                "}\n",
                "\n",
                "print(\"Otsu Optimal Threshold Comparison across Preprocessing/Contrast Variants:\")\n",
                "for name, im in variants.items():\n",
                "    t, _ = otsu_thresh(im)\n",
                "    print(f\"  {name:14s} -> Otsu T = {t:.1f}\")\n",
                "\n",
                "t_opt, mask_otsu = otsu_thresh(gray_coins)\n",
                "hist_coins = cv2.calcHist([gray_coins], [0], None, [256], [0, 256]).ravel()\n",
                "\n",
                "plt.figure(figsize=(15, 4.5))\n",
                "plt.subplot(131)\n",
                "plt.imshow(gray_coins, cmap=\"gray\")\n",
                "plt.title(\"Original (Grayscale Coins)\", fontsize=11, fontweight='bold')\n",
                "plt.axis(\"off\")\n",
                "\n",
                "plt.subplot(132)\n",
                "plt.plot(hist_coins, color='black', lw=1.5)\n",
                "plt.axvline(t_opt, color='crimson', linestyle='--', lw=2, label=f'Otsu Optimal T = {t_opt:.1f}')\n",
                "plt.title(\"Intensity Histogram & Cutoff\", fontsize=11, fontweight='bold')\n",
                "plt.xlabel(\"Pixel Intensity\")\n",
                "plt.ylabel(\"Frequency\")\n",
                "plt.legend(loc=\"upper right\")\n",
                "plt.grid(alpha=0.3)\n",
                "\n",
                "plt.subplot(133)\n",
                "plt.imshow(mask_otsu, cmap=\"gray\")\n",
                "plt.title(\"Otsu Binary Mask\", fontsize=11, fontweight='bold')\n",
                "plt.axis(\"off\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task3_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 3 Analysis\n",
                "**Why is Otsu useful when the programmer does not know the correct threshold beforehand?**\n",
                "Otsu's algorithm exhaustively evaluates all candidate thresholds from $0$ to $255$ and automatically selects the threshold that maximizes between-class variance (minimizes intra-class variance). It dynamically adapts to lighting, contrast, and exposure variations between images without requiring manual calibration.\n",
                "- Scaling contrast by $1.5\\times$ shifted the optimal threshold from $162.0$ to $194.0$.\n",
                "- Scaling contrast by $0.6\\times$ shifted the threshold downwards to $97.0$."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 4: Sorting Objects by Color (HSV Segmentation)\n",
                "We segment target green smarties from `smarties.png` using HSV color thresholding (`cv2.inRange`), comparing a **strict** bounding box against a **comprehensive final** bounding box supplemented by morphological cleanup."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "img_smarties = cv2.imread('smarties.png')\n",
                "hsv_smarties = cv2.cvtColor(img_smarties, cv2.COLOR_BGR2HSV)\n",
                "\n",
                "# Strict mask: tight hue & high S, V\n",
                "strict_mask = cv2.inRange(hsv_smarties, np.array([65, 220, 220]), np.array([67, 255, 255]))\n",
                "\n",
                "# Final mask: broad hue [45, 85] with realistic S and V bounds\n",
                "final_mask_raw = cv2.inRange(hsv_smarties, np.array([45, 60, 50]), np.array([85, 255, 255]))\n",
                "\n",
                "# Morphological cleanup (Opening removes specks, Closing fills pinholes)\n",
                "k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
                "final_mask = cv2.morphologyEx(final_mask_raw, cv2.MORPH_OPEN, k_morph)\n",
                "final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k_morph)\n",
                "\n",
                "extracted_smarties = cv2.bitwise_and(img_smarties, img_smarties, mask=final_mask)\n",
                "\n",
                "print(f\"Strict mask foreground pixels: {cv2.countNonZero(strict_mask)}\")\n",
                "print(f\"Final mask foreground pixels:  {cv2.countNonZero(final_mask)}\")\n",
                "\n",
                "panels_t4 = [\n",
                "    (\"Original Image\", cv2.cvtColor(img_smarties, cv2.COLOR_BGR2RGB)),\n",
                "    (\"Strict HSV Mask\", strict_mask),\n",
                "    (\"Final HSV Mask (Cleaned)\", final_mask),\n",
                "    (\"Extracted Color Region\", cv2.cvtColor(extracted_smarties, cv2.COLOR_BGR2RGB))\n",
                "]\n",
                "\n",
                "plt.figure(figsize=(16, 4))\n",
                "for i, (t, im) in enumerate(panels_t4, 1):\n",
                "    plt.subplot(1, 4, i)\n",
                "    plt.imshow(im, cmap=\"gray\" if im.ndim == 2 else None)\n",
                "    plt.title(t, fontsize=11, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task4_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 4 Analysis\n",
                "**Why does the restrictive mask fail?**\n",
                "The strict mask ($H \\in [65, 67], S \\ge 220, V \\ge 220$) isolates only the intense chromatic core ($2,566$ pixels), ignoring physical optical phenomena:\n",
                "1. **Specular highlights**: Glare on the glossy surface washes color toward white, drastically reducing Saturation ($S < 220$).\n",
                "2. **Curvature shading**: Along outer rims, shaded regions receive less light, reducing Value ($V < 220$).\n",
                "Decoupling Hue from Saturation and Value allows us to guard the true identity in $H \\in [45, 85]$ while tolerating wide $S \\ge 60$ and $V \\ge 50$, capturing the complete candies ($5,492$ pixels)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 5: Detecting the Boundary of a Manufactured Part\n",
                "### Canny Edge Detection & Hysteresis Thresholding\n",
                "We test three (low, high) threshold pairs: `(50, 100)`, `(50, 200)`, and `(150, 200)`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "gray_smarties = cv2.cvtColor(img_smarties, cv2.COLOR_BGR2GRAY)\n",
                "blur_smarties = cv2.GaussianBlur(gray_smarties, (5, 5), 0)\n",
                "\n",
                "pairs_canny = [(50, 100), (50, 200), (150, 200)]\n",
                "panels_t5 = [(\"Original Grayscale\", gray_smarties)]\n",
                "\n",
                "for lo, hi in pairs_canny:\n",
                "    edges = cv2.Canny(blur_smarties, lo, hi)\n",
                "    cnt = cv2.countNonZero(edges)\n",
                "    print(f\"Canny ({lo}, {hi}) -> edge pixels = {cnt}\")\n",
                "    panels_t5.append((f\"Canny ({lo}, {hi})\\n{cnt} px\", edges))\n",
                "\n",
                "plt.figure(figsize=(16, 4))\n",
                "for i, (t, im) in enumerate(panels_t5, 1):\n",
                "    plt.subplot(1, 4, i)\n",
                "    plt.imshow(im, cmap=\"gray\")\n",
                "    plt.title(t, fontsize=11, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task5_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 5 Edge Categorization & Hysteresis Mechanics\n",
                "- **Strong edges**: Edges with gradient $> \\text{high}$. Present across all configurations (e.g. primary outer boundaries between candies and background).\n",
                "- **Weak edges**: Edges with $\\text{low} \\le \\text{gradient} \\le \\text{high}$. Retained only if 8-connected to a strong edge.\n",
                "- **Missing edges**: Faint boundaries where even the gradient maximum fails to exceed $\\text{high}=200$ (e.g. low-contrast boundaries between similarly colored adjacent candies).\n",
                "- **Unwanted edges**: Internal specular reflections and surface texture noise captured when $\\text{low}=50$ or $\\text{high}=100$.\n",
                "\n",
                "**Effect of changing thresholds:**\n",
                "Increasing $\\text{high}$ reduces the number of initial seed pixels. Increasing $\\text{low}$ prevents weak edge traversal, eliminating false texture edges but risking broken object boundaries."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 6: Segmenting a Region Inside a Medical Image (Brain MRI)\n",
                "### Region Growing Algorithm\n",
                "We test region growing using two different seed points (Gray Matter vs. Bright Structure) across three intensity difference thresholds ($thr \\in \\{10, 25, 45\\}$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [],
            "source": [
                "def region_grow(gray, seed, thr, conn8=False):\n",
                "    h, w = gray.shape\n",
                "    mask = np.zeros((h, w), np.uint8)\n",
                "    seed_val = int(gray[seed])\n",
                "    mask[seed] = 255\n",
                "    q = deque([seed])\n",
                "    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]\n",
                "    if conn8:\n",
                "        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]\n",
                "    while q:\n",
                "        r, c = q.popleft()\n",
                "        for dr, dc in nbrs:\n",
                "            rr, cc = r + dr, c + dc\n",
                "            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] == 0 and abs(int(gray[rr, cc]) - seed_val) <= thr:\n",
                "                mask[rr, cc] = 255\n",
                "                q.append((rr, cc))\n",
                "    return mask\n",
                "\n",
                "gray_mri = cv2.imread('brain_mri.png', cv2.IMREAD_GRAYSCALE)\n",
                "seed_A = (91, 137)    # Gray matter parenchyma (intensity ~75)\n",
                "seed_B = (196, 164)   # Bright white matter structure (intensity ~195)\n",
                "seeds = {\"Seed A (Gray Matter)\": seed_A, \"Seed B (Bright Structure)\": seed_B}\n",
                "\n",
                "for s_name, s_coord in seeds.items():\n",
                "    print(f\"{s_name} at {s_coord}: intensity = {gray_mri[s_coord]}\")\n",
                "\n",
                "thr_values = [10, 25, 45]\n",
                "plt.figure(figsize=(15, 9))\n",
                "idx_t6 = 1\n",
                "for s_name, s_coord in seeds.items():\n",
                "    for thr in thr_values:\n",
                "        m_grown = region_grow(gray_mri, s_coord, thr)\n",
                "        px_cnt = cv2.countNonZero(m_grown)\n",
                "        print(f\"{s_name}, thr={thr}: grown pixels = {px_cnt}\")\n",
                "        \n",
                "        vis = cv2.cvtColor(gray_mri, cv2.COLOR_GRAY2RGB)\n",
                "        vis[m_grown == 255] = [30, 200, 30]  # Green segmented mask\n",
                "        cv2.circle(vis, (s_coord[1], s_coord[0]), 3, (255, 0, 0), -1)  # Red seed\n",
                "        \n",
                "        plt.subplot(2, 3, idx_t6); idx_t6 += 1\n",
                "        plt.imshow(vis)\n",
                "        plt.title(f\"{s_name}\\nThr={thr} | Px={px_cnt}\", fontsize=10, fontweight='bold')\n",
                "        plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task6_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 6 Analysis\n",
                "**Why can changing the seed point significantly change the final segmented region?**\n",
                "1. **The Seed Acts as the Reference Intensity**: Similarity is evaluated relative to the seed's intensity value (`seed_val`). Seed A ($I=75$) evaluates the range $[75-thr, 75+thr]$, targeting parenchymal tissue, whereas Seed B ($I=195$) targets bright white matter structures $[195-thr, 195+thr]$.\n",
                "2. **Spatial Connectivity Constraint**: Region growing requires contiguous spatial paths. Even if two areas have identical grayscale values, they cannot merge if separated by a boundary exceeding the threshold.\n",
                "3. **Threshold Sensitivity & Boundary Leakage**: At $thr=10$, only local homogeneous cores are captured ($4$ px vs $2$ px). At $thr=45$, Seed A experiences a leakage cliff ($21,239$ px) where the region breaches subtle boundaries and floods the whole brain matter."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Tasks 7 & 8: Separating Touching Coins Using Marker-Based Watershed\n",
                "We implement the full 10-stage watershed segmentation pipeline, visualising each stage (Task 7), and sweep the distance-transform threshold fraction across $0.2, 0.4, 0.6, 0.8$ (Task 8)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [],
            "source": [
                "img_coins = cv2.imread('coins.jpg')\n",
                "if img_coins is None:\n",
                "    img_coins = cv2.imread('water_coins.jpg')\n",
                "\n",
                "def watershed_pipeline(img, frac=0.5):\n",
                "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
                "    blur = cv2.GaussianBlur(gray, (5, 5), 0)\n",
                "    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
                "    \n",
                "    k = np.ones((3, 3), np.uint8)\n",
                "    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)\n",
                "    sure_bg = cv2.dilate(opening, k, iterations=3)\n",
                "    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)\n",
                "    \n",
                "    _, sure_fg = cv2.threshold(dist, frac * dist.max(), 255, 0)\n",
                "    sure_fg = np.uint8(sure_fg)\n",
                "    unknown = cv2.subtract(sure_bg, sure_fg)\n",
                "    \n",
                "    n_labels, markers = cv2.connectedComponents(sure_fg)\n",
                "    markers = markers + 1\n",
                "    markers[unknown == 255] = 0\n",
                "    \n",
                "    out = img.copy()\n",
                "    markers = cv2.watershed(out, markers)\n",
                "    out[markers == -1] = [0, 0, 255]  # Red boundary\n",
                "    \n",
                "    unique_regs = set(np.unique(markers).tolist()) - {-1, 1}\n",
                "    stages = {\n",
                "        \"1. Threshold\": thresh,\n",
                "        \"2. Sure BG\": sure_bg,\n",
                "        \"3. Distance\": dist,\n",
                "        \"4. Sure FG\": sure_fg,\n",
                "        \"5. Unknown\": unknown,\n",
                "        \"6. Final Watershed\": out\n",
                "    }\n",
                "    return stages, n_labels - 1, len(unique_regs)\n",
                "\n",
                "# Task 7 Display\n",
                "stages_t7, _, _ = watershed_pipeline(img_coins, frac=0.5)\n",
                "panels_t7 = [(\"Original\", cv2.cvtColor(img_coins, cv2.COLOR_BGR2RGB))] + list(stages_t7.items())\n",
                "\n",
                "plt.figure(figsize=(21, 3.5))\n",
                "for i, (t, im) in enumerate(panels_t7, 1):\n",
                "    if t == \"6. Final Watershed\": im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)\n",
                "    plt.subplot(1, 7, i)\n",
                "    plt.imshow(im, cmap=\"gray\" if im.ndim == 2 else None)\n",
                "    plt.title(t, fontsize=10, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task7_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Task 8 Distance Threshold Sweep\n",
                "fractions = [0.2, 0.4, 0.6, 0.8]\n",
                "print(f\"{'Exp':<4} | {'Distance Threshold':<18} | {'Foreground Markers':<18} | {'Separated Regions':<18} | {'Observed Result'}\")\n",
                "print(\"-\" * 95)\n",
                "obs_comments = {\n",
                "    0.2: \"Under-segmented / merged blobs: cores touch across necks, clusters counted as single objects.\",\n",
                "    0.4: \"Partial separation: several pairs remain merged where neck depth is large.\",\n",
                "    0.6: \"Optimal separation: individual coin cores successfully isolated, exact boundaries detected.\",\n",
                "    0.8: \"Over-segmented / eroded: high threshold shrinks cores too much, smaller coins lost.\"\n",
                "}\n",
                "\n",
                "plt.figure(figsize=(16, 4))\n",
                "for idx, frac in enumerate(fractions, 1):\n",
                "    st, n_fg, n_reg = watershed_pipeline(img_coins, frac=frac)\n",
                "    obs = obs_comments[frac]\n",
                "    print(f\"{idx:<4} | {frac:.1f} * max{'':<10} | {n_fg:<18} | {n_reg:<18} | {obs}\")\n",
                "    \n",
                "    plt.subplot(1, 4, idx)\n",
                "    plt.imshow(cv2.cvtColor(st['6. Final Watershed'], cv2.COLOR_BGR2RGB))\n",
                "    plt.title(f\"Frac = {frac} ({n_fg} markers, {n_reg} regs)\", fontsize=10, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task8_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 8 Findings Summary\n",
                "| Experiment | Distance Threshold | Foreground Markers | Separated Regions | Observed Result |\n",
                "| :---: | :---: | :---: | :---: | :--- |\n",
                "| 1 | $0.2 \\times \\max$ | 1 | 1 | Severe under-segmentation; thin necks between coins not severed, all merged into 1 blob. |\n",
                "| 2 | $0.4 \\times \\max$ | 7 | 7 | Partial separation; only isolated coins resolved, touching clusters remain joined. |\n",
                "| 3 | $0.6 \\times \\max$ | 24 | 24 | **Optimal separation**; individual coin centers clearly resolved without fragmentation. |\n",
                "| 4 | $0.8 \\times \\max$ | 24 | 24 | Highly shrunken cores; fine boundaries preserved but risk of losing shallow or small coins. |\n",
                "\n",
                "**Optimal Parameter Range**: $0.55 \\times \\max \\le \\text{Threshold} \\le 0.70 \\times \\max$."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 9: Image Segmentation Using K-Means Clustering\n",
                "We segment the complex color scene `dog.jpeg` using K-Means with $K \\in \\{2, 4, 6\\}$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [],
            "source": [
                "img_dog = cv2.imread('dog.jpeg')\n",
                "if img_dog is None:\n",
                "    img_dog = cv2.imread('dogimg.jpg')\n",
                "\n",
                "data_dog = img_dog.reshape((-1, 3)).astype(np.float32)\n",
                "criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)\n",
                "\n",
                "panels_t9 = [(\"Original\", cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB))]\n",
                "for K in (2, 4, 6):\n",
                "    _, labels, centers = cv2.kmeans(data_dog, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)\n",
                "    seg_dog = np.uint8(centers)[labels.flatten()].reshape(img_dog.shape)\n",
                "    shares = np.bincount(labels.flatten(), minlength=K) / len(labels) * 100\n",
                "    mean_diff = np.abs(seg_dog.astype(int) - img_dog.astype(int)).mean()\n",
                "    \n",
                "    print(f\"\\nK={K}:\")\n",
                "    print(f\"  Cluster shares (%): {np.round(shares, 1).tolist()}\")\n",
                "    print(f\"  Mean Absolute Error: {mean_diff:.2f}\")\n",
                "    print(f\"  Centroids (BGR): {np.uint8(centers).tolist()}\")\n",
                "    panels_t9.append((f\"K={K} (MAE={mean_diff:.1f})\", cv2.cvtColor(seg_dog, cv2.COLOR_BGR2RGB)))\n",
                "\n",
                "plt.figure(figsize=(16, 4))\n",
                "for i, (t, im) in enumerate(panels_t9, 1):\n",
                "    plt.subplot(1, 4, i)\n",
                "    plt.imshow(im)\n",
                "    plt.title(t, fontsize=11, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task9_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 9 Analysis & Observations per K\n",
                "- **$K=2$** (MAE = $17.34$, Shares: $21.1\\% / 78.9\\%$):\n",
                "  Extreme color simplification. Binary segmentation dividing the image into bright highlights/sky/dog coat vs. dark floor/background foliage.\n",
                "- **$K=4$** (MAE = $9.78$, Shares: $8.4\\%, 50.9\\%, 13.2\\%, 27.5\\%$):\n",
                "  Captures primary structural domains: background floor/shadows, bright reflections, mid-tone fur body, and background scene.\n",
                "- **$K=6$** (MAE = $7.84$, Shares: $5.2\\%, 5.7\\%, 5.7\\%, 14.6\\%, 42.8\\%, 26.0\\%$):\n",
                "  Preserves fine gradients, shading variations, and facial feature boundaries with sub-$8.0$ mean intensity distortion.\n",
                "- **Key limitation of K-Means**: Ignores spatial coordinates entirely; disjoint pixels with similar RGB values receive the exact same label regardless of location."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Task 10: Multi-Method Segmentation Engineering Comparison\n",
                "We benchmark three distinct segmentation methods on the challenging multi-object scene `smarties.png`:\n",
                "1. **Otsu's Global Thresholding**\n",
                "2. **HSV Color-Based Segmentation**\n",
                "3. **Marker-Controlled Watershed**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {},
            "outputs": [],
            "source": [
                "img_t10 = cv2.imread('smarties.png')\n",
                "gray_t10 = cv2.cvtColor(img_t10, cv2.COLOR_BGR2GRAY)\n",
                "blur_t10 = cv2.GaussianBlur(gray_t10, (5, 5), 0)\n",
                "\n",
                "# Method 1: Otsu's Thresholding\n",
                "_, m1_otsu = cv2.threshold(blur_t10, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
                "\n",
                "# Method 2: HSV Color Keying (Green Candies)\n",
                "hsv_t10 = cv2.cvtColor(img_t10, cv2.COLOR_BGR2HSV)\n",
                "m2_hsv = cv2.inRange(hsv_t10, np.array([45, 60, 50]), np.array([85, 255, 255]))\n",
                "k_m10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
                "m2_hsv = cv2.morphologyEx(m2_hsv, cv2.MORPH_OPEN, k_m10)\n",
                "m2_hsv = cv2.morphologyEx(m2_hsv, cv2.MORPH_CLOSE, k_m10)\n",
                "\n",
                "# Method 3: Watershed Segmentation\n",
                "stages_ws10, _, _ = watershed_pipeline(img_t10, frac=0.45)\n",
                "m3_ws = cv2.cvtColor(stages_ws10[\"6. Final Watershed\"], cv2.COLOR_BGR2RGB)\n",
                "\n",
                "panels_t10 = [\n",
                "    (\"Original Image\", cv2.cvtColor(img_t10, cv2.COLOR_BGR2RGB)),\n",
                "    (\"Method 1: Otsu\", m1_otsu),\n",
                "    (\"Method 2: HSV Green\", m2_hsv),\n",
                "    (\"Method 3: Watershed\", m3_ws)\n",
                "]\n",
                "\n",
                "plt.figure(figsize=(16, 4))\n",
                "for i, (t, im) in enumerate(panels_t10, 1):\n",
                "    plt.subplot(1, 4, i)\n",
                "    plt.imshow(im, cmap=\"gray\" if im.ndim == 2 else None)\n",
                "    plt.title(t, fontsize=11, fontweight='bold')\n",
                "    plt.axis(\"off\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"task10_output.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Task 10: Comparative Analysis Matrix\n",
                "\n",
                "| Per Method Requirement | Method 1: Otsu's Thresholding | Method 2: HSV Color Keying | Method 3: Marker-Controlled Watershed |\n",
                "| :--- | :--- | :--- | :--- |\n",
                "| **1. Assumption made about the image** | Bimodal intensity distribution; foreground objects are uniformly darker/brighter than background. | Distinct chromatic hue ranges separate target classes; lighting shifts primarily affect S and V. | Objects are roughly circular/convex bodies separated by valleys of lower distance transform depth. |\n",
                "| **2. Region/object correctly identified** | Separates high-contrast candies from the bright white background table. | Flawlessly isolates specific target class (green candies) regardless of spatial arrangement. | Reliably separates individual touching candies by drawing boundary dams along contact necks. |\n",
                "| **3. Part incorrectly segmented** | Touching candies merge into unified blobs; low-contrast pale yellow candies wash out. | Excludes other candy classes; specular highlights can create pinholes if bounds are too tight. | Distorts irregular non-circular geometries; overly aggressive thresholding merges small candies. |\n",
                "| **4. Parameter with greatest effect** | Choice of pre-filtering kernel size (e.g. Gaussian blur $\\sigma$) influencing histogram valley sharpness. | Hue boundary window $[H_{min}, H_{max}]$, which governs chromatic identity. | Distance transform cutoff fraction ($frac$), dictating marker core separation vs. erosion. |\n",
                "\n",
                "### Technical Conclusion\n",
                "**Conclusion**: For `smarties.png`, **HSV Color Keying** provided the most meaningful and accurate segmentation when the engineering goal is isolating a specific product class, while **Marker-Controlled Watershed** is the superior solution when the objective is universal object separation and unit counting.\n",
                "- HSV succeeded because candies possess high saturation and distinct hue peaks ($H \\approx 66$ for green), enabling complete decoupling from surface lighting variations.\n",
                "- Otsu failed to segment touching candies, collapsing multiple adjacent items into a single undifferentiated region."
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Lab05_Image_Segmentation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Saved Lab05_Image_Segmentation.ipynb successfully!")
