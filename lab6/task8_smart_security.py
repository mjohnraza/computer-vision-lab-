"""
Task 8: Smart Security System Using Boundary Detection
======================================================
Technique: Polygonal ROI Masking + Gaussian Pre-filtering + Canny Boundary Detection +
           Empirical Baseline Edge Statistical Calibration + Consecutive-Frame Streak Alarm Logic.

Problem Statement:
- Monitor a sensitive restricted physical zone in a continuous video surveillance stream.
- Detect boundaries of foreign/unauthorized objects placed within or encroaching into the zone.
- Trigger an immediate automated security alarm while maintaining robustness against single-frame
  flicker, minor lighting shifts, and sensor noise.

Parameters chosen and rationale:
- Predefined Security Zone Polygon: [(200, 150), (500, 150), (500, 400), (200, 400)].
  Directs surveillance resources strictly to high-value assets/restricted space.
- GaussianBlur(5, 5, sigma=0): Prevents high-frequency camera CMOS sensor noise from registering as edges.
- Canny(50, 150): Detects sharp structural object boundaries (ratios 1:3).
- Baseline Window (N = 30 frames):
  Calibrates background ambient edge density (mean mu and standard deviation sigma) when zone is empty.
- Dynamic Threshold limit = mu + 4 * sigma + 50:
  Provides a robust statistical buffer exceeding 99.99% of ambient noise fluctuations.
- Temporal Streak Filter (consecutive_needed = 4):
  Requires the edge count to remain continuously above threshold for at least 4 consecutive frames
  before raising the alarm, filtering out transient glitches, insect flight, or brief shadows.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_security_system(video_path="assets/security_video.mp4",
                        output_video_path="task8_security_alarm.mp4",
                        output_plot_path="task8_security_output.png",
                        baseline_frames=30,
                        consecutive_needed=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Predefined restricted zone polygon (x, y)
    zone = np.array([[(200, 150), (500, 150), (500, 400), (200, 400)]], np.int32)
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, zone, 255)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    def get_zone_edges(frame_bgr):
        g = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(g, 50, 150)
        z_edges = cv2.bitwise_and(edges, zone_mask)
        return z_edges, cv2.countNonZero(z_edges)

    # 1. Baseline Calibration Phase
    print(f"Calibrating baseline edge density over initial {baseline_frames} frames...")
    baseline_counts = []
    frames_cache = []

    for f_idx in range(baseline_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_cache.append(frame.copy())
        _, count = get_zone_edges(frame)
        baseline_counts.append(count)

    mu_base = float(np.mean(baseline_counts)) if baseline_counts else 0.0
    std_base = float(np.std(baseline_counts)) if baseline_counts else 0.0
    limit = mu_base + 4.0 * std_base + 50.0

    print(f" - Baseline Mean: {mu_base:.2f} edge pixels")
    print(f" - Baseline Std:  {std_base:.2f}")
    print(f" - Alarm Trigger Threshold: {limit:.2f} edge pixels")

    # 2. Surveillance & Intrusion Detection Phase
    frame_idx = 0
    streak = 0
    alarm_triggered_count = 0
    history_counts = []
    history_alarms = []

    secure_snapshot = None
    intrusion_snapshot = None
    intrusion_edge_snapshot = None

    # Write calibrated baseline frames
    for fr in frames_cache:
        frame_idx += 1
        cnt = baseline_counts[frame_idx - 1]
        history_counts.append(cnt)
        history_alarms.append(False)

        cv2.polylines(fr, zone, True, (0, 255, 0), 2)
        cv2.putText(fr, "SECURITY STATUS: NORMAL", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(fr, f"Calibrating Frame {frame_idx}/{baseline_frames}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        writer.write(fr)
        if frame_idx == 15:
            secure_snapshot = fr.copy()

    # Process remaining surveillance frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        z_edges, cnt = get_zone_edges(frame)
        history_counts.append(cnt)

        if cnt > limit:
            streak += 1
        else:
            streak = 0

        alarm_active = (streak >= consecutive_needed)
        history_alarms.append(alarm_active)

        annotated = frame.copy()
        if alarm_active:
            alarm_triggered_count += 1
            # Zone border turns RED
            cv2.polylines(annotated, zone, True, (0, 0, 255), 3)

            # Highlight bounding box of intrusive edges
            pts = cv2.findNonZero(z_edges)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 140, 255), 2)

            # Warning banner
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(annotated, "ALARM: UNAUTHORIZED OBJECT DETECTED", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(annotated, f"Zone Edge Count: {cnt} (Threshold: {limit:.0f})", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if intrusion_snapshot is None and streak == consecutive_needed:
                intrusion_snapshot = annotated.copy()
                intrusion_edge_snapshot = z_edges.copy()
        else:
            cv2.polylines(annotated, zone, True, (0, 255, 0), 2)
            cv2.putText(annotated, "SECURITY STATUS: NORMAL", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Zone Edge Count: {cnt} (Limit: {limit:.0f})", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(annotated)

    cap.release()
    writer.release()

    # 3. Generate Multi-Panel Analysis Figure
    fig = plt.figure(figsize=(15, 10))

    # Panel 1: Secure state
    plt.subplot(2, 2, 1)
    if secure_snapshot is not None:
        plt.imshow(cv2.cvtColor(secure_snapshot, cv2.COLOR_BGR2RGB))
    plt.title("1. Baseline Phase (Zone Empty - Normal)", fontsize=11, fontweight="bold")
    plt.axis("off")

    # Panel 2: Intrusion state
    plt.subplot(2, 2, 2)
    if intrusion_snapshot is not None:
        plt.imshow(cv2.cvtColor(intrusion_snapshot, cv2.COLOR_BGR2RGB))
    plt.title("2. Intrusion Detected (Alarm Triggered)", fontsize=11, fontweight="bold")
    plt.axis("off")

    # Panel 3: Edge map inside restricted zone
    plt.subplot(2, 2, 3)
    if intrusion_edge_snapshot is not None:
        plt.imshow(intrusion_edge_snapshot, cmap="hot")
    plt.title("3. Canny Boundaries of Intruding Object", fontsize=11, fontweight="bold")
    plt.axis("off")

    # Panel 4: Temporal Timeline Plot of Edge Counts vs Threshold
    plt.subplot(2, 2, 4)
    time_steps = np.arange(1, len(history_counts) + 1)
    plt.plot(time_steps, history_counts, color="#1f77b4", linewidth=1.8, label="Zone Edge Pixels")
    plt.axhline(limit, color="red", linestyle="--", linewidth=1.5, label=f"Alarm Limit ({limit:.0f} px)")
    plt.axvline(baseline_frames, color="gray", linestyle=":", label="End Baseline")

    alarm_frames = np.where(history_alarms)[0] + 1
    if len(alarm_frames) > 0:
        plt.scatter(alarm_frames, [history_counts[i-1] for i in alarm_frames],
                    color="red", s=30, zorder=4, label="Active Alarm")

    plt.title("4. Real-Time Boundary Count vs Alarm Threshold", fontsize=11, fontweight="bold")
    plt.xlabel("Video Frame", fontsize=10)
    plt.ylabel("Non-Zero Edge Pixels in Zone", fontsize=10)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Task 8 Complete:")
    print(f" - Total Video Frames: {frame_idx}")
    print(f" - Alarm Triggered Frames: {alarm_triggered_count}")
    print(f" - Saved security video: {output_video_path}")
    print(f" - Saved security summary plot: {output_plot_path}")

    return {
        "frames": frame_idx,
        "alarm_frames": alarm_triggered_count,
        "limit": limit
    }

if __name__ == "__main__":
    run_security_system()
