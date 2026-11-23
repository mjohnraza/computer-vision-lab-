"""
Task 0: Data Exploration & Ground-Truth Bounding Box Inspection
===============================================================
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 08)

Matches sections 4.1, 4.2, 4.3, 4.4 of the Lab Manual:
- Loads annotations.csv
- Checks class distributions and image counts
- Plots ground-truth bounding boxes on sample images with custom colors
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    if rel_path.startswith("lab8/") and os.path.exists(rel_path[5:]):
        return rel_path[5:]
    if os.path.exists(os.path.join("lab8", rel_path)):
        return os.path.join("lab8", rel_path)
    if os.path.basename(os.getcwd()) == "lab8" and rel_path.startswith("lab8/"):
        return rel_path[5:]
    return rel_path

output_dir = "outputs" if os.path.basename(os.getcwd()) == "lab8" else "lab8/outputs"
data_dir = "data" if os.path.basename(os.getcwd()) == "lab8" else "lab8/data"
os.makedirs(output_dir, exist_ok=True)

def explore_data():
    csv_path = get_path("lab8/data/annotations.csv")
    train = pd.read_csv(csv_path)

    print("=================================================================")
    print("      LAB 08 - TASK 0: DATA EXPLORATION & ANNOTATION INSPECTION  ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")
    print("\n[*] First 5 rows of annotations (train.head()):")
    print(train.head().to_string())

    print(f"\n[*] Total Bounding Box Annotations: {len(train)}")
    print(f"[*] Number of unique training images: {train['image_names'].nunique()}")

    print("\n[*] Class Distribution (value_counts):")
    counts = train['cell_type'].value_counts()
    print(counts.to_string())

    # Visualizing Ground-Truth Bounding Boxes (matching Section 4.4 of Manual)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    color_map = {
        'Car': ('#1f77b4', 'blue'),
        'Pedestrian': ('#ff7f0e', 'orange'),
        'Traffic_Sign': ('#2ca02c', 'green'),
        'Bicycle': ('#d62728', 'red')
    }

    sample_images = ["1.jpg", "2.jpg"]
    train_img_dir = os.path.join(data_dir, "train_images")

    for ax, img_name in zip(axes, sample_images):
        img_full_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"Image: {img_name} - Ground-Truth Annotations", fontsize=12, fontweight='bold')

        # Filter annotations for this image
        img_anns = train[train['image_names'] == img_name]

        for _, row in img_anns.iterrows():
            xmin = row['xmin']
            xmax = row['xmax']
            ymin = row['ymin']
            ymax = row['ymax']
            width = xmax - xmin
            height = ymax - ymin
            ctype = row['cell_type']

            edgecolor, textcol = color_map.get(ctype, ('yellow', 'yellow'))

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2.5,
                                     edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)

            ax.text(xmin, max(0, ymin - 6), ctype, color='white', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor=edgecolor, alpha=0.85))

        ax.axis('on')
        ax.grid(False)

    plt.suptitle("Lab 08: Autonomous Traffic Dataset - Ground-Truth Bounding Boxes\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(output_dir, "data_exploration_gt.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[*] Saved ground-truth visualization to {out_fig}")
    return out_fig

if __name__ == "__main__":
    explore_data()
