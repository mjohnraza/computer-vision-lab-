"""
Task 3: R-CNN End-to-End Inference Pipeline with NMS
====================================================
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 08)

R-CNN Step 3:
- Loads trained R-CNN model (ResNet18 backbone + heads).
- Runs Selective Search on unseen test images to propose candidates.
- Warps and normalizes candidate crops.
- Evaluates classifier and bounding box regressor.
- Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections.
- Plots and saves high-resolution object detection results.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
from task2_rcnn_training import RCNNModel

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

class_names = ['Background', 'Car', 'Pedestrian', 'Traffic_Sign']
color_palette = {
    'Car': '#1f77b4',          # Blue
    'Pedestrian': '#ff7f0e',   # Orange
    'Traffic_Sign': '#2ca02c', # Green
}

def extract_test_proposals(img, max_proposals=300):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    h, w = img.shape[:2]
    proposals = []
    for x, y, rw, rh in rects[:max_proposals]:
        if rw < 20 or rh < 20:
            continue
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(w - 1, x + rw)
        ymax = min(h - 1, y + rh)
        proposals.append([xmin, ymin, xmax, ymax])
    return proposals

def run_rcnn_inference_on_image(img, model, device, conf_thresh=0.65, nms_thresh=0.30):
    """
    Runs full R-CNN detection pipeline on a single image:
    1. Selective Search
    2. Warp to 224x224 & Normalize
    3. Batched CNN Forward Pass
    4. Bounding Box Delta Refinement
    5. NMS Filtering
    """
    proposals = extract_test_proposals(img, max_proposals=250)
    if not proposals:
        return []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    batch_tensors = []
    valid_proposals = []

    for prop in proposals:
        px1, py1, px2, py2 = prop
        crop = img[py1:py2, px1:px2]
        if crop.size == 0 or (px2 - px1) < 15 or (py2 - py1) < 15:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_warped = cv2.resize(crop_rgb, (224, 224))
        tensor = transform(crop_warped)
        batch_tensors.append(tensor)
        valid_proposals.append(prop)

    if not batch_tensors:
        return []

    input_batch = torch.stack(batch_tensors).to(device)

    model.eval()
    with torch.no_grad():
        cls_scores, bbox_deltas = model(input_batch)
        probs = torch.softmax(cls_scores, dim=1).cpu().numpy()
        deltas = bbox_deltas.cpu().numpy()

    detections = []
    # For each class (excluding Background 0)
    for cls_idx in range(1, len(class_names)):
        cls_name = class_names[cls_idx]
        cls_boxes = []
        cls_scores_list = []

        for i, (prop, prob) in enumerate(zip(valid_proposals, probs)):
            score = prob[cls_idx]
            if score >= conf_thresh:
                # Apply bounding box delta regression
                px1, py1, px2, py2 = prop
                pw = px2 - px1
                ph = py2 - py1
                pcx = px1 + pw / 2.0
                pcy = py1 + ph / 2.0

                tx, ty, tw, th = deltas[i]
                # Clamp deltas to prevent explosive expansions
                tx = np.clip(tx, -1.0, 1.0)
                ty = np.clip(ty, -1.0, 1.0)
                tw = np.clip(tw, -1.5, 1.5)
                th = np.clip(th, -1.5, 1.5)

                pred_cx = pcx + pw * tx
                pred_cy = pcy + ph * ty
                pred_w = pw * np.exp(tw)
                pred_h = ph * np.exp(th)

                rx1 = max(0, pred_cx - pred_w / 2.0)
                ry1 = max(0, pred_cy - pred_h / 2.0)
                rx2 = min(img.shape[1] - 1, pred_cx + pred_w / 2.0)
                ry2 = min(img.shape[0] - 1, pred_cy + pred_h / 2.0)

                cls_boxes.append([rx1, ry1, rx2, ry2])
                cls_scores_list.append(score)

        if cls_boxes:
            boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(cls_scores_list, dtype=torch.float32)
            keep = ops.nms(boxes_tensor, scores_tensor, nms_thresh)

            for k in keep:
                b = cls_boxes[k]
                sc = cls_scores_list[k]
                detections.append({
                    "class": cls_name,
                    "score": float(sc),
                    "box": b
                })

    return detections

def evaluate_test_set():
    print("=================================================================")
    print("      LAB 08 - TASK 3: R-CNN INFERENCE & OBJECT DETECTION       ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_path("lab8/outputs/rcnn_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = RCNNModel(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Successfully loaded trained R-CNN weights.")

    test_dir = os.path.join(data_dir, "test_images")
    test_files = sorted(os.listdir(test_dir))[:4]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, test_file in zip(axes.flat, test_files):
        img_path = os.path.join(test_dir, test_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = run_rcnn_inference_on_image(img, model, device, conf_thresh=0.55, nms_thresh=0.25)
        print(f"[*] Test Image {test_file}: Detected {len(detections)} objects.")
        for d in detections:
            print(f"    - {d['class']} ({d['score']*100:.1f}%) at {d['box']}")

        ax.imshow(img_rgb)
        ax.set_title(f"Test Image: {test_file} | R-CNN Predictions ({len(detections)} detected)",
                     fontsize=12, fontweight='bold')

        for d in detections:
            x1, y1, x2, y2 = d['box']
            cls = d['class']
            score = d['score']
            color = color_palette.get(cls, '#ffff00')

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            label_txt = f"{cls}: {score*100:.1f}%"
            ax.text(x1, max(0, y1 - 6), label_txt, color='white', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor=color, alpha=0.85))

        ax.axis('off')

    plt.suptitle("Lab 08: R-CNN Object Detection on Autonomous Driving Test Dataset\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    result_fig = os.path.join(output_dir, "rcnn_detection_results.png")
    plt.savefig(result_fig, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[*] Saved R-CNN object detection predictions to {result_fig}")
    return result_fig

if __name__ == "__main__":
    evaluate_test_set()
