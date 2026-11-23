"""
Dataset Generator for Lab 08: Autonomous Traffic Object Detection (R-CNN)
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence

Creates:
- data/train_images/ and data/test_images/
- data/annotations.csv with columns: [image_names, cell_type, xmin, xmax, ymin, ymax]
Classes: 'Car', 'Pedestrian', 'Traffic_Sign', 'Bicycle'
"""

import os
import cv2
import numpy as np
import pandas as pd

def create_road_scene(img_idx, width=640, height=480, is_test=False):
    # Canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Background Buildings / Trees
    for y in range(0, int(height * 0.45)):
        img[y, :] = [int(220 - 40 * y / (height * 0.45)), int(200 - 30 * y / (height * 0.45)), 170]

    # Buildings silhouette in background
    np.random.seed(img_idx * 17 + (100 if is_test else 0))
    x_b = 0
    while x_b < width:
        bw = np.random.randint(50, 110)
        bh = np.random.randint(100, 180)
        bcol = (np.random.randint(60, 100), np.random.randint(70, 110), np.random.randint(80, 120))
        cv2.rectangle(img, (x_b, int(height * 0.45) - bh), (min(width, x_b + bw), int(height * 0.45)), bcol, -1)
        # Windows
        for wy in range(int(height * 0.45) - bh + 15, int(height * 0.45) - 15, 25):
            for wx in range(x_b + 10, min(width, x_b + bw) - 10, 20):
                if np.random.rand() > 0.3:
                    cv2.rectangle(img, (wx, wy), (wx + 10, wy + 15), (200, 240, 255), -1)
        x_b += bw + 5

    # 2. Road & Sidewalk
    road_top = int(height * 0.45)
    # Sidewalk
    cv2.rectangle(img, (0, road_top), (width, road_top + 45), (140, 140, 140), -1)
    cv2.line(img, (0, road_top + 45), (width, road_top + 45), (100, 100, 100), 2)
    # Asphalt
    cv2.rectangle(img, (0, road_top + 45), (width, height), (55, 55, 60), -1)
    # Lane divider dashes
    for lx in range(20, width, 60):
        cv2.line(img, (lx, int(height * 0.75)), (lx + 35, int(height * 0.75)), (255, 255, 255), 4)

    # 3. Add Objects: Car, Pedestrian, Traffic_Sign, Bicycle
    annotations = []

    # Decide layout based on random seed
    num_cars = np.random.randint(1, 3)
    num_peds = np.random.randint(1, 3)
    num_signs = np.random.randint(1, 2)

    # Draw Traffic Signs (on sidewalk / roadside poles)
    for s_i in range(num_signs):
        sx = np.random.randint(40, width - 60)
        sy = road_top - np.random.randint(40, 90)
        # Pole
        cv2.line(img, (sx + 15, sy + 30), (sx + 15, road_top + 35), (120, 120, 130), 3)
        # Sign face: Octagon (Stop sign) or Circle (Speed Limit)
        sign_type = np.random.choice(["stop", "speed"])
        if sign_type == "stop":
            cv2.circle(img, (sx + 15, sy + 15), 18, (0, 0, 210), -1)
            cv2.circle(img, (sx + 15, sy + 15), 18, (255, 255, 255), 2)
            cv2.putText(img, "STOP", (sx + 3, sy + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.circle(img, (sx + 15, sy + 15), 18, (255, 255, 255), -1)
            cv2.circle(img, (sx + 15, sy + 15), 18, (0, 0, 210), 3)
            cv2.putText(img, "50", (sx + 8, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        annotations.append({
            "cell_type": "Traffic_Sign",
            "xmin": max(0, sx - 2),
            "ymin": max(0, sy - 2),
            "xmax": min(width - 1, sx + 32),
            "ymax": min(height - 1, sy + 35)
        })

    # Draw Pedestrians (on sidewalk or zebra crossing)
    for p_i in range(num_peds):
        px = np.random.randint(30, width - 50)
        py = road_top + np.random.randint(2, 25)
        ped_h = np.random.randint(55, 75)
        ped_w = int(ped_h * 0.38)
        shirt_col = (np.random.randint(40, 220), np.random.randint(40, 220), np.random.randint(40, 220))
        pants_col = (np.random.randint(20, 60), np.random.randint(20, 60), np.random.randint(60, 120))

        # Head
        cv2.circle(img, (px + ped_w // 2, py + int(ped_h * 0.15)), int(ped_h * 0.12), (180, 205, 225), -1)
        # Torso
        cv2.rectangle(img, (px, py + int(ped_h * 0.28)), (px + ped_w, py + int(ped_h * 0.65)), shirt_col, -1)
        # Legs
        cv2.line(img, (px + int(ped_w * 0.3), py + int(ped_h * 0.65)), (px + int(ped_w * 0.2), py + ped_h), pants_col, 3)
        cv2.line(img, (px + int(ped_w * 0.7), py + int(ped_h * 0.65)), (px + int(ped_w * 0.8), py + ped_h), pants_col, 3)

        annotations.append({
            "cell_type": "Pedestrian",
            "xmin": max(0, px - 2),
            "ymin": max(0, py),
            "xmax": min(width - 1, px + ped_w + 3),
            "ymax": min(height - 1, py + ped_h + 2)
        })

    # Draw Cars (on asphalt road)
    placed_x = []
    for c_i in range(num_cars):
        # Find non-overlapping X
        cx = np.random.randint(40, width - 180)
        cy = road_top + 50 + c_i * 90 + np.random.randint(5, 30)
        car_w = np.random.randint(130, 170)
        car_h = int(car_w * 0.52)
        car_col = (np.random.randint(30, 220), np.random.randint(30, 220), np.random.randint(30, 220))

        # Car lower body
        cv2.rectangle(img, (cx, cy + int(car_h * 0.4)), (cx + car_w, cy + car_h), car_col, -1)
        cv2.rectangle(img, (cx, cy + int(car_h * 0.4)), (cx + car_w, cy + car_h), (20, 20, 20), 2)
        # Car cabin / roof
        roof_pts = np.array([
            [cx + int(car_w * 0.2), cy + int(car_h * 0.4)],
            [cx + int(car_w * 0.35), cy],
            [cx + int(car_w * 0.75), cy],
            [cx + int(car_w * 0.88), cy + int(car_h * 0.4)]
        ], np.int32)
        cv2.fillPoly(img, [roof_pts], car_col)
        cv2.polylines(img, [roof_pts], isClosed=True, color=(20, 20, 20), thickness=2)

        # Windows
        win_pts = np.array([
            [cx + int(car_w * 0.25), cy + int(car_h * 0.38)],
            [cx + int(car_w * 0.37), cy + 4],
            [cx + int(car_w * 0.72), cy + 4],
            [cx + int(car_w * 0.82), cy + int(car_h * 0.38)]
        ], np.int32)
        cv2.fillPoly(img, [win_pts], (220, 240, 250))

        # Wheels
        r = int(car_h * 0.22)
        cv2.circle(img, (cx + int(car_w * 0.25), cy + car_h), r, (30, 30, 30), -1)
        cv2.circle(img, (cx + int(car_w * 0.25), cy + car_h), int(r * 0.5), (180, 180, 180), -1)
        cv2.circle(img, (cx + int(car_w * 0.78), cy + car_h), r, (30, 30, 30), -1)
        cv2.circle(img, (cx + int(car_w * 0.78), cy + car_h), int(r * 0.5), (180, 180, 180), -1)

        # Headlights / Taillights
        cv2.circle(img, (cx + car_w - 4, cy + int(car_h * 0.55)), 6, (0, 240, 255), -1)
        cv2.circle(img, (cx + 4, cy + int(car_h * 0.55)), 6, (0, 0, 240), -1)

        annotations.append({
            "cell_type": "Car",
            "xmin": max(0, cx - 2),
            "ymin": max(0, cy),
            "xmax": min(width - 1, cx + car_w + 3),
            "ymax": min(height - 1, cy + car_h + r)
        })

    return img, annotations

def generate_full_dataset(num_train=30, num_test=8):
    print("[*] Generating Traffic Object Detection dataset for Lab 08...")
    os.makedirs("lab8/data/train_images", exist_ok=True)
    os.makedirs("lab8/data/test_images", exist_ok=True)
    os.makedirs("lab8/outputs", exist_ok=True)

    records = []

    # Generate training images
    for i in range(1, num_train + 1):
        filename = f"{i}.jpg"
        img, anns = create_road_scene(i, is_test=False)
        cv2.imwrite(f"lab8/data/train_images/{filename}", img)

        for ann in anns:
            records.append({
                "image_names": filename,
                "cell_type": ann["cell_type"],
                "xmin": ann["xmin"],
                "xmax": ann["xmax"],
                "ymin": ann["ymin"],
                "ymax": ann["ymax"]
            })

    # Save annotations CSV
    df = pd.DataFrame(records)
    df.to_csv("lab8/data/annotations.csv", index=False)
    print(f"  -> Generated {num_train} train images in lab8/data/train_images/")
    print(f"  -> Saved annotations.csv with {len(df)} bounding boxes.")
    print("  -> Class distribution:")
    print(df["cell_type"].value_counts().to_string())

    # Generate test images
    for j in range(1, num_test + 1):
        test_filename = f"{j}.jpg"
        test_img, _ = create_road_scene(j + 200, is_test=True)
        cv2.imwrite(f"lab8/data/test_images/{test_filename}", test_img)
    print(f"  -> Generated {num_test} test images in lab8/data/test_images/")

if __name__ == "__main__":
    generate_full_dataset()
