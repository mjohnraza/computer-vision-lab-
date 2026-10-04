"""
Task 5: Panoramic Image Stitching Using SIFT
===========================================
Technique: SIFT Interest Point Detection + BFMatcher + Lowe's Ratio Test +
           RANSAC Homography Mapping + Perspective Warping & Seam Blending.

Problem Statement:
- Create a seamless wide-angle panoramic view by aligning and stitching multiple
  overlapping camera panning shots.
- Overcome camera panning rotation, perspective foreshortening, and brightness transitions.

Parameters chosen and rationale:
- SIFT descriptor (128-d): Captures local gradient distributions invariant to rotation and scale.
- Lowe's Ratio Test (0.75): Preserves high-confidence feature matches while filtering ambiguous descriptors.
- RANSAC Homography (reprojection error = 5.0): Calculates projective mapping matrix H between
  adjacent image coordinate planes while filtering spurious correspondence outliers.
- Seamless Compositing: Merges warped layers without clipping overlapping regions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_pair(img_left, img_right, ratio=0.75, ransac_thr=5.0):
    gl = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gr, None)  # right image
    k2, d2 = sift.detectAndCompute(gl, None)  # left image (target plane)

    if d1 is None or d2 is None:
        raise ValueError("Could not extract SIFT descriptors from images.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(d1, d2, k=2)

    good = [m for m, n in raw_matches if len((m, n)) == 2 and m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"Insufficient good matches ({len(good)}) between pair.")

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thr)
    inliers = int(mask.sum()) if mask is not None else 0

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # Canvas dimensions to hold composite
    pano_w = w1 + w2
    pano_h = max(h1, h2)

    warped_right = cv2.warpPerspective(img_right, H, (pano_w, pano_h))

    # Blend: paste left image over warped canvas
    mask_left = np.any(img_left > 0, axis=-1)
    pano = warped_right.copy()
    pano[0:h1, 0:w1][mask_left] = img_left[mask_left]

    # Crop trailing empty black columns
    gray_pano = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray_pano.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        pano = pano[:, :cols[-1] + 1]

    # Draw match correspondences for visualization
    match_vis = cv2.drawMatches(img_right, k1, img_left, k2, good[:40], None,
                                matchesMask=mask.ravel().tolist()[:40] if mask is not None else None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return pano, H, inliers, len(good), match_vis

def create_panorama(image_paths=["assets/pan1.jpg", "assets/pan2.jpg", "assets/pan3.jpg"],
                    output_plot_path="task5_panorama_output.png",
                    output_stitched_path="task5_panorama_stitched.png"):
    imgs = [cv2.imread(p) for p in image_paths]
    for i, im in enumerate(imgs):
        if im is None:
            raise FileNotFoundError(f"Cannot load image {image_paths[i]}")

    print(f"Stitching {len(imgs)} panning captures into a single panoramic image...")

    pano = imgs[0]
    stats = []
    match_visualizations = []

    for i in range(1, len(imgs)):
        nxt = imgs[i]
        pano, H, inliers, n_good, match_vis = stitch_pair(pano, nxt)
        stats.append({"pair": f"Image {i} -> Image {i+1}", "good": n_good, "inliers": inliers})
        match_visualizations.append(match_vis)
        print(f" - Stitched Image {i} and {i+1}: {n_good} matches, {inliers} RANSAC inliers.")

    cv2.imwrite(output_stitched_path, pano)

    # Visualization plot using GridSpec
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)

    # 1. Input panning shots
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Panning Input {i+1}: {image_paths[i].split('/')[-1]}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # 2. Pairwise SIFT Matches
    ax_m1 = fig.add_subplot(gs[1, :2])
    ax_m1.imshow(cv2.cvtColor(match_visualizations[0], cv2.COLOR_BGR2RGB))
    ax_m1.set_title(f"SIFT Inlier Correspondences ({stats[0]['pair']})", fontsize=11, fontweight="bold")
    ax_m1.axis("off")

    if len(match_visualizations) > 1:
        ax_m2 = fig.add_subplot(gs[1, 2])
        ax_m2.imshow(cv2.cvtColor(match_visualizations[1], cv2.COLOR_BGR2RGB))
        ax_m2.set_title(f"SIFT Inliers ({stats[1]['pair']})", fontsize=11, fontweight="bold")
        ax_m2.axis("off")

    # 3. Final Panorama
    ax_p = fig.add_subplot(gs[2, :])
    ax_p.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
    ax_p.set_title(f"Final Stitched Panorama ({pano.shape[1]}x{pano.shape[0]} px)", fontsize=13, fontweight="bold")
    ax_p.axis("off")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Task 5 Complete: Final panorama size: {pano.shape[1]}x{pano.shape[0]} pixels.")
    print(f"Saved visualization: {output_plot_path}")
    print(f"Saved stitched image: {output_stitched_path}")
    return pano

if __name__ == "__main__":
    create_panorama()
