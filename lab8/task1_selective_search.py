"""
Task 1: Region Proposal Generation Using Selective Search & IoU Matching
========================================================================
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 08)

R-CNN Step 1:
- Applies Selective Search to extract ~1000-2000 candidate object regions.
- Computes Intersection-over-Union (IoU) with Ground Truth annotations.
- Classifies proposals into positive object classes (IoU >= 0.5) and background (IoU < 0.2).
- Prepares warped cropped regions (224x224) for CNN training.
"""

import os
import cv2
import numpy as np
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

def compute_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes:
    box = [xmin, ymin, xmax, ymax]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def extract_selective_search_proposals(img, max_proposals=500):
    """
    Executes OpenCV Selective Search segmentation in Fast mode.
    Returns list of proposals: [xmin, ymin, xmax, ymax]
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposals = []
    h, w = img.shape[:2]
    for x, y, rw, rh in rects[:max_proposals]:
        # Filter tiny noise boxes or edge degenerate rectangles
        if rw < 15 or rh < 15:
            continue
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(w - 1, x + rw)
        ymax = min(h - 1, y + rh)
        proposals.append([xmin, ymin, xmax, ymax])

    return proposals

def generate_rcnn_training_samples(max_images=25, positive_iou_thresh=0.50, negative_iou_thresh=0.15):
    """
    Iterates through dataset, extracts proposals via Selective Search,
    matches with ground-truth boxes, and collects warped 224x224 crops.
    """
    print("\n[*] Generating R-CNN Region Proposals and Training Dataset...")
    csv_path = get_path("lab8/data/annotations.csv")
    train_df = pd.read_csv(csv_path)
    train_img_dir = os.path.join(data_dir, "train_images")

    all_crops = []
    all_labels = []
    all_bboxes = [] # [xmin, ymin, xmax, ymax]
    all_gt_deltas = [] # (tx, ty, tw, th) bbox regression targets

    class_names = ['Background', 'Car', 'Pedestrian', 'Traffic_Sign']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    unique_images = train_df['image_names'].unique()[:max_images]

    pos_count = 0
    neg_count = 0

    for idx, img_name in enumerate(unique_images):
        img_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_boxes = []
        gt_classes = []
        for _, row in train_df[train_df['image_names'] == img_name].iterrows():
            gt_boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            gt_classes.append(row['cell_type'])

        proposals = extract_selective_search_proposals(img, max_proposals=400)

        # Match proposals with ground-truth
        image_pos = 0
        image_neg = 0

        for prop in proposals:
            px1, py1, px2, py2 = prop
            pw = px2 - px1
            ph = py2 - py1
            pcx = px1 + pw / 2.0
            pcy = py1 + ph / 2.0

            best_iou = 0.0
            best_gt_class = None
            best_gt_box = None

            for gt_box, gt_cls in zip(gt_boxes, gt_classes):
                iou = compute_iou(prop, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_class = gt_cls
                    best_gt_box = gt_box

            # Positive Sample (IoU >= threshold)
            if best_iou >= positive_iou_thresh and image_pos < 12:
                crop = img[py1:py2, px1:px2]
                if crop.size > 0:
                    crop_warped = cv2.resize(crop, (224, 224))
                    all_crops.append(crop_warped)
                    all_labels.append(class_to_idx[best_gt_class])
                    all_bboxes.append(prop)

                    # Compute bounding box regression targets
                    gx1, gy1, gx2, gy2 = best_gt_box
                    gw = gx2 - gx1
                    gh = gy2 - gy1
                    gcx = gx1 + gw / 2.0
                    gcy = gy1 + gh / 2.0
                    tx = (gcx - pcx) / float(pw)
                    ty = (gcy - pcy) / float(ph)
                    tw = np.log(max(1e-4, gw / float(pw)))
                    th = np.log(max(1e-4, gh / float(ph)))
                    all_gt_deltas.append([tx, ty, tw, th])

                    image_pos += 1
                    pos_count += 1

            # Negative Sample (Background, low IoU)
            elif best_iou < negative_iou_thresh and image_neg < 15:
                crop = img[py1:py2, px1:px2]
                if crop.size > 0:
                    crop_warped = cv2.resize(crop, (224, 224))
                    all_crops.append(crop_warped)
                    all_labels.append(class_to_idx['Background'])
                    all_bboxes.append(prop)
                    all_gt_deltas.append([0.0, 0.0, 0.0, 0.0]) # no offset for background
                    image_neg += 1
                    neg_count += 1

        # Also add ground-truth boxes directly as guaranteed high-quality positives
        for gt_box, gt_cls in zip(gt_boxes, gt_classes):
            gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
            crop = img[gy1:gy2, gx1:gx2]
            if crop.size > 0:
                crop_warped = cv2.resize(crop, (224, 224))
                all_crops.append(crop_warped)
                all_labels.append(class_to_idx[gt_cls])
                all_bboxes.append(gt_box)
                all_gt_deltas.append([0.0, 0.0, 0.0, 0.0])
                pos_count += 1

    all_crops = np.array(all_crops, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_gt_deltas = np.array(all_gt_deltas, dtype=np.float32)

    print(f"[*] Total R-CNN proposals generated: {len(all_crops)}")
    print(f"    - Positive object samples: {pos_count}")
    print(f"    - Negative background samples: {neg_count}")

    # Save dataset to compressed NPZ
    npz_path = os.path.join(output_dir, "rcnn_proposals_dataset.npz")
    np.savez_compressed(npz_path, crops=all_crops, labels=all_labels, deltas=all_gt_deltas)
    print(f"[*] Saved proposals dataset to {npz_path}")

    # Visualize Selective Search on a sample image
    sample_img_path = os.path.join(train_img_dir, "1.jpg")
    s_img = cv2.imread(sample_img_path)
    sample_props = extract_selective_search_proposals(s_img, max_proposals=80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image with Selective Search Region Proposals (Top 80)", fontsize=11, fontweight='bold')
    for p in sample_props:
        rect = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], linewidth=1.2,
                                 edgecolor='#00ffcc', facecolor='none', alpha=0.6)
        axes[0].add_patch(rect)
    axes[0].axis('off')

    # Ground truth vs matched positive proposals
    axes[1].imshow(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("IoU Matched Proposals (Green: Positives >= 0.5 IoU, Red: GT)", fontsize=11, fontweight='bold')

    # Draw GT
    for _, row in train_df[train_df['image_names'] == "1.jpg"].iterrows():
        grect = patches.Rectangle((row['xmin'], row['ymin']), row['xmax']-row['xmin'], row['ymax']-row['ymin'],
                                  linewidth=2.5, edgecolor='red', facecolor='none')
        axes[1].add_patch(grect)
        axes[1].text(row['xmin'], row['ymin'] - 5, row['cell_type'], color='white',
                     bbox=dict(facecolor='red', alpha=0.8, pad=0.2), fontsize=9, fontweight='bold')

    # Draw high IoU proposals
    for p in sample_props:
        for _, row in train_df[train_df['image_names'] == "1.jpg"].iterrows():
            gt_b = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            iou = compute_iou(p, gt_b)
            if iou >= 0.50:
                prect = patches.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], linewidth=2.0,
                                          edgecolor='#00ff00', facecolor='none', linestyle='--')
                axes[1].add_patch(prect)

    axes[1].axis('off')
    plt.suptitle("R-CNN: Selective Search Region Proposals & Ground-Truth IoU Matching\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    vis_path = os.path.join(output_dir, "selective_search_proposals.png")
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[*] Saved proposal visualization to {vis_path}")
    return npz_path, vis_path

if __name__ == "__main__":
    generate_rcnn_training_samples()
