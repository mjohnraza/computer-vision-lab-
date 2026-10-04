"""
Master Runner Script for Lab 06:
Executes all 8 tasks and Foundation Task, verifying end-to-end functionality.
"""
import time
import os
import sys

def main():
    print("=" * 70)
    print("      FAST NUCES - COMPUTER VISION LAB 06: MASTER RUNNER")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Ensure assets exist
    print("\n[Step 1/10] Verifying and generating assets...")
    import generate_all_assets
    generate_all_assets.make_edges_input()
    generate_all_assets.make_lab_screens()
    generate_all_assets.make_asset_tracking_data()
    generate_all_assets.make_video_object_data()
    generate_all_assets.make_panorama_data()
    generate_all_assets.make_road_data()
    generate_all_assets.make_security_video()

    # Step 2: Foundation Task (Sobel, Canny, LoG)
    print("\n[Step 2/10] Running Foundation Task (Sobel, Canny, LoG)...")
    import task0_edges
    task0_edges.run_task0()

    # Step 3: Task 1 - Screen Detection
    print("\n[Step 3/10] Running Task 1: Computer Screen Detection (Hough Lines)...")
    import task1_screen_detection
    screens = task1_screen_detection.detect_screens()

    # Step 4: Task 2 - Asset Tracking
    print("\n[Step 4/10] Running Task 2: Asset Tracking in Computer Lab (SIFT)...")
    import task2_asset_tracking
    inventory = task2_asset_tracking.track_assets()

    # Step 5: Task 3 - Wavelet Anomaly Detection
    print("\n[Step 5/10] Running Task 3: Anomaly Detection in Sensor Data (Wavelet)...")
    import task3_wavelet_anomaly
    anomalies = task3_wavelet_anomaly.run_wavelet_anomaly_detection()

    # Step 6: Task 4 - Object Recognition in Video
    print("\n[Step 6/10] Running Task 4: Object Recognition in Video (SIFT + Homography)...")
    import task4_object_recognition
    recog = task4_object_recognition.run_object_recognition()

    # Step 7: Task 5 - Panorama Stitching
    print("\n[Step 7/10] Running Task 5: Panoramic Image Stitching (SIFT + Homography)...")
    import task5_panorama
    pano = task5_panorama.create_panorama()

    # Step 8: Task 6 - Lane Detection
    print("\n[Step 8/10] Running Task 6: Autonomous Lane Detection (ROI + Hough Lines)...")
    import task6_lane_detection
    lanes = task6_lane_detection.detect_lanes()

    # Step 9: Task 7 - Coin Detection and Counting
    print("\n[Step 9/10] Running Task 7: Coin Detection & Counting (Hough Circles)...")
    import task7_coins_counting
    coins = task7_coins_counting.count_coins()

    # Step 10: Task 8 - Smart Security System
    print("\n[Step 10/10] Running Task 8: Smart Security System (Zone Canny + Baseline)...")
    import task8_smart_security
    sec = task8_smart_security.run_security_system()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("                    ALL LAB 06 TASKS COMPLETED!")
    print(f"               Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 70)
    print("\nGenerated Output Files:")
    outputs = [
        "task0_edges_output.png",
        "task1_screens_output.png", "task1_screens_detected.png",
        "task2_asset_tracking_output.png", "task2_tracked_scene.png",
        "task3_wavelet_anomaly.png",
        "task4_recognition_output.png", "task4_recognized_video.mp4",
        "task5_panorama_output.png", "task5_panorama_stitched.png",
        "task6_lane_output.png", "task6_lane_detected.png",
        "task7_coins_output.png", "task7_coins_detected.png",
        "task8_security_output.png", "task8_security_alarm.mp4"
    ]
    for out in outputs:
        status = "EXISTS" if os.path.exists(out) else "MISSING"
        size = f"{os.path.getsize(out) / 1024:.1f} KB" if status == "EXISTS" else "N/A"
        print(f"  [{status}] {out:<32} ({size})")
    print("=" * 70)

if __name__ == "__main__":
    main()
