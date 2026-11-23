"""
Lab 09: Modern Object Detection & Real-Time Pose Estimation
Task 2: Comparative Object Detection using YOLOv3, SSD, and RetinaNet
Author: Muhammad John Raza (Roll No: 23k-0069)
FAST NUCES Karachi, Department of Artificial Intelligence
Course: Computer Vision (AI-4002) | Instructor: Talha Shahid

Requirements:
1. Implement YOLOv3, SSD, and RetinaNet on a common image/video dataset.
2. Use pre-trained models available in OpenCV, TensorFlow, or PyTorch.
3. Display bounding boxes, class labels, and confidence scores for detected objects.
4. Save comparative visualizations side by side.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models.detection as detection
import matplotlib.pyplot as plt
from ultralytics import YOLO

def get_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("lab9") / rel_path
    if p2.exists():
        return p2
    if Path("lab9").is_dir():
        return Path("lab9") / rel_path
    return p1

# Standard COCO 91-category mapping used by torchvision models
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ModelSuite:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"[INFO] Initializing Object Detection Suite on {self.device}...")
        
        # 1. YOLOv3
        yolov3_path = get_path("models/yolov3-tinyu.pt")
        if not yolov3_path.exists():
            yolov3_path = "yolov3-tinyu.pt"
        print(f"[INFO] Loading YOLOv3 ({yolov3_path})...")
        self.yolo_model = YOLO(str(yolov3_path))

        # 2. SSD MobileNetV3 Large
        print("[INFO] Loading SSDLite320-MobileNetV3...")
        self.ssd_model = detection.ssdlite320_mobilenet_v3_large(
            weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        ).to(self.device).eval()

        # 3. RetinaNet ResNet50 FPN
        print("[INFO] Loading RetinaNet-ResNet50-FPN...")
        self.retinanet_model = detection.retinanet_resnet50_fpn(
            weights=detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device).eval()

        self.transform = T.Compose([T.ToTensor()])

    def detect_yolov3(self, image_bgr, conf_thresh=0.30):
        t0 = time.time()
        results = self.yolo_model.predict(source=image_bgr, conf=conf_thresh, verbose=False)[0]
        latency = (time.time() - t0) * 1000.0

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.yolo_model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    'box': [int(v) for v in xyxy],
                    'label': label,
                    'score': conf
                })
        return detections, latency

    def detect_torchvision(self, model, image_bgr, conf_thresh=0.30):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            preds = model([tensor])[0]
        latency = (time.time() - t0) * 1000.0

        detections = []
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()

        for box, score, label_id in zip(boxes, scores, labels):
            if score >= conf_thresh:
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id] if label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else f"cls_{label_id}"
                if label_name != '__background__' and label_name != 'N/A':
                    detections.append({
                        'box': [int(v) for v in box],
                        'label': label_name,
                        'score': float(score)
                    })
        return detections, latency

def annotate_image(image_bgr, detections, model_name, color=(0, 255, 0), latency_ms=0.0):
    vis = image_bgr.copy()
    
    # Draw header banner
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (20, 20, 25), -1)
    banner_text = f"{model_name} | Detections: {len(detections)} | Latency: {latency_ms:.1f} ms"
    cv2.putText(vis, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        score = det['score']

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f"{label}: {score*100:.1f}%"
        (w_txt, h_txt), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(y1 - 22, 0)), (x1 + w_txt + 6, max(y1, 22)), color, -1)
        cv2.putText(vis, tag, (x1 + 3, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis

def run_comparative_detection():
    suite = ModelSuite(device='cpu')
    output_dir = get_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        ("Desk Setup (Laptop, Keyboard, Mouse)", get_path("assets/lab_assets_01.jpg")),
        ("Lab Workstation with Student", get_path("assets/lab_assets_02.jpg")),
        ("Multi-Asset Desktop Bench", get_path("assets/lab_assets_03.jpg")),
    ]

    print("\n--- Running Comparative Detection (YOLOv3 vs. SSD vs. RetinaNet) ---")
    all_results = {}

    for title, img_path in test_images:
        print(f"\nProcessing Image: {img_path.name} ({title})...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not load {img_path}, skipping.")
            continue

        # 1. YOLOv3
        yolo_dets, yolo_lat = suite.detect_yolov3(img, conf_thresh=0.25)
        # 2. SSD
        ssd_dets, ssd_lat = suite.detect_torchvision(suite.ssd_model, img, conf_thresh=0.25)
        # 3. RetinaNet
        retina_dets, retina_lat = suite.detect_torchvision(suite.retinanet_model, img, conf_thresh=0.25)

        print(f"  -> YOLOv3:    {len(yolo_dets):2d} objects ({yolo_lat:6.1f} ms) | {[d['label'] for d in yolo_dets[:4]]}")
        print(f"  -> SSD:       {len(ssd_dets):2d} objects ({ssd_lat:6.1f} ms) | {[d['label'] for d in ssd_dets[:4]]}")
        print(f"  -> RetinaNet: {len(retina_dets):2d} objects ({retina_lat:6.1f} ms) | {[d['label'] for d in retina_dets[:4]]}")

        # Visualizations
        vis_yolo = annotate_image(img, yolo_dets, "YOLOv3", color=(0, 255, 128), latency_ms=yolo_lat)
        vis_ssd = annotate_image(img, ssd_dets, "SSD MobileNetV3", color=(0, 165, 255), latency_ms=ssd_lat)
        vis_retina = annotate_image(img, retina_dets, "RetinaNet ResNet50", color=(255, 255, 0), latency_ms=retina_lat)

        stem = img_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_yolov3.jpg"), vis_yolo)
        cv2.imwrite(str(output_dir / f"{stem}_ssd.jpg"), vis_ssd)
        cv2.imwrite(str(output_dir / f"{stem}_retinanet.jpg"), vis_retina)

        all_results[stem] = {
            'yolo': (vis_yolo, len(yolo_dets), yolo_lat),
            'ssd': (vis_ssd, len(ssd_dets), ssd_lat),
            'retina': (vis_retina, len(retina_dets), retina_lat),
        }

    # Generate Grand 3x3 Comparison Grid
    print("\nGenerating 3x3 Comparative Visualization Grid...")
    fig, axes = plt.subplots(len(test_images), 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for row_idx, (title, img_path) in enumerate(test_images):
        stem = img_path.stem
        if stem not in all_results:
            continue

        res = all_results[stem]
        models = [
            ("YOLOv3-tiny", res['yolo'][0], res['yolo'][1], res['yolo'][2]),
            ("SSD-MobileNetV3", res['ssd'][0], res['ssd'][1], res['ssd'][2]),
            ("RetinaNet-ResNet50", res['retina'][0], res['retina'][1], res['retina'][2]),
        ]

        for col_idx, (m_name, vis_bgr, num_dets, lat) in enumerate(models):
            ax = axes[row_idx, col_idx]
            rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"{m_name} on {stem}\n({num_dets} detected, {lat:.1f} ms)", fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.suptitle("Lab 09: Comparative Object Detection (YOLOv3 vs. SSD vs. RetinaNet)\nStudent: Muhammad John Raza (23k-0069) | Instructor: Talha Shahid", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    grid_out = output_dir / "comparison_yolov3_ssd_retinanet.png"
    plt.savefig(str(grid_out), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved 3x3 comparison grid: {grid_out}")

if __name__ == "__main__":
    run_comparative_detection()
