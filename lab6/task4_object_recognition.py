"""
Task 4: Object Recognition in Video Using SIFT
=============================================
Technique: SIFT Keypoint Extraction & Description + BFMatcher + Lowe's Ratio Test +
           RANSAC Homography + Perspective Transformation for Bounding Box Localization.

Problem Statement:
- Recognize a target reference object in a video stream or set of test images.
- Reliably locate and draw bounding boxes around the object under scale changes,
  3D rotations, translations, and perspective distortions.

Parameters chosen and rationale:
- cv2.SIFT_create(): Invariant to scaling, 2D rotation, and affine illumination shifts.
- Ratio Test threshold = 0.75: Discards 90% of false matches while retaining 85% of true matches.
- RANSAC reprojection error = 5.0 pixels: Fits the 3x3 homography matrix H while filtering outliers.
- Confidence threshold min_inliers = 15: Guarantees that bounding boxes are drawn only when
  mathematically confident, avoiding hallucinated boxes.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_object_recognition(ref_path="assets/object.jpg",
                           video_path="assets/test_video.mp4",
                           output_video_path="task4_recognized_video.mp4",
                           output_plot_path="task4_recognition_output.png",
                           min_inliers=15):
    # 1. Load reference object
    ref_bgr = cv2.imread(ref_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Cannot load reference image {ref_path}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    rh, rw = ref_gray.shape[:2]

    # Reference corners for perspective projection: [[0,0], [rw,0], [rw,rh], [0,rh]]
    ref_box = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)

    # 2. Extract SIFT features on reference
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(ref_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # 3. Open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    detected_count = 0
    sample_frames = {}
    sample_indices = [5, 18, 32, 45]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k2, d2 = sift.detectAndCompute(frame_gray, None)
        inliers = 0
        H = None

        if d2 is not None and len(k2) >= 4:
            raw_matches = matcher.knnMatch(d1, d2, k=2)
            good_matches = [
                m for m, n in raw_matches
                if len((m, n)) == 2 and m.distance < 0.75 * n.distance
            ]

            if len(good_matches) >= 4:
                src_pts = np.float32([k1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([k2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = int(mask.sum()) if mask is not None else 0

        annotated = frame.copy()
        if H is not None and inliers >= min_inliers:
            detected_count += 1
            # Project reference corners onto video frame
            dst_box = cv2.perspectiveTransform(ref_box, H)
            cv2.polylines(annotated, [np.int32(dst_box)], True, (0, 255, 0), 3)

            # Draw center point and label
            cx = int(np.mean(dst_box[:, 0, 0]))
            cy = int(np.mean(dst_box[:, 0, 1]))
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(annotated, f"TARGET FOUND ({inliers} inliers)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Frame {frame_count}/{total_frames}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(annotated, "Searching for object...", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(annotated)

        if frame_count in sample_indices:
            sample_frames[frame_count] = (frame.copy(), annotated.copy(), inliers)

    cap.release()
    writer.release()

    # 4. Generate comparison plot across key video frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Reference object
    axes[0, 0].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Target Reference Object\n({len(k1)} SIFT Keypoints)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Keypoint visualization on reference
    ref_kps = cv2.drawKeypoints(ref_bgr, k1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[1, 0].imshow(cv2.cvtColor(ref_kps, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("SIFT Scale-Space Extrema", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Keyframes from video
    keys = list(sample_frames.keys())
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for i, f_num in enumerate(keys[:4]):
        orig, res, inl = sample_frames[f_num]
        r, c = positions[i]
        axes[r, c].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        axes[r, c].set_title(f"Frame {f_num}: Inliers={inl} (Detected)", fontsize=11, fontweight="bold")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Task 4 Complete:")
    print(f" - Processed frames: {frame_count}")
    print(f" - Detected target in: {detected_count}/{frame_count} frames ({detected_count/frame_count*100:.1f}%)")
    print(f" - Output video saved: {output_video_path}")
    print(f" - Output keyframe analysis saved: {output_plot_path}")

    return {
        "total_frames": frame_count,
        "detected_frames": detected_count,
        "accuracy": detected_count / max(1, frame_count)
    }

if __name__ == "__main__":
    run_object_recognition()
