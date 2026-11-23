"""
Lab 09: Modern Object Detection & Real-Time Pose Estimation
Task 3: Benchmarking Detection Speed (FPS), Confidence, Model Size & Latency
Author: Muhammad John Raza (Roll No: 23k-0069)
FAST NUCES Karachi, Department of Artificial Intelligence
Course: Computer Vision (AI-4002) | Instructor: Talha Shahid

Requirements:
1. Measure Detection Speed (FPS) for YOLOv3, SSD, and RetinaNet.
2. Measure Accuracy / Confidence Scores on different object types.
3. Record Model Size (MB & Parameter Count) and Inference Time per frame (ms).
4. Save benchmark metrics to CSV and generate publication-quality visual comparison plots.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
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

def count_parameters(model):
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    return 0

def benchmark_models(num_warmup=1, num_runs=2):
    output_dir = get_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Staging Models for Comprehensive Benchmarks ---")
    
    # 1. YOLOv3-tiny
    yolo_file = get_path("models/yolov3-tinyu.pt")
    yolo_model = YOLO(str(yolo_file))
    yolo_size_mb = os.path.getsize(str(yolo_file)) / (1024 * 1024)
    yolo_params = sum(p.numel() for p in yolo_model.model.parameters())

    # 2. SSD MobileNetV3
    ssd_model = detection.ssdlite320_mobilenet_v3_large(
        weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    ssd_params = count_parameters(ssd_model)
    # 3.4M float32 params ≈ 13.4 MB
    ssd_size_mb = (ssd_params * 4) / (1024 * 1024)

    # 3. RetinaNet ResNet50 FPN
    retinanet_model = detection.retinanet_resnet50_fpn(
        weights=detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    ).eval()
    retina_params = count_parameters(retinanet_model)
    # 34.0M float32 params ≈ 130.0 MB
    retina_size_mb = (retina_params * 4) / (1024 * 1024)

    models_info = {
        'YOLOv3-tiny': {
            'model': yolo_model,
            'type': 'ultralytics',
            'params': yolo_params,
            'size_mb': yolo_size_mb,
            'color': '#2ca02c', # Green
        },
        'SSD-MobileNetV3': {
            'model': ssd_model,
            'type': 'torchvision',
            'params': ssd_params,
            'size_mb': ssd_size_mb,
            'color': '#ff7f0e', # Orange
        },
        'RetinaNet-ResNet50': {
            'model': retinanet_model,
            'type': 'torchvision',
            'params': retina_params,
            'size_mb': retina_size_mb,
            'color': '#1f77b4', # Blue
        }
    }

    test_imgs = [
        get_path("assets/lab_assets_01.jpg"),
        get_path("assets/lab_assets_02.jpg"),
        get_path("assets/lab_assets_03.jpg")
    ]
    loaded_imgs = [cv2.imread(str(p)) for p in test_imgs if p.exists()]
    transform = T.Compose([T.ToTensor()])

    print(f"[INFO] Loaded {len(loaded_imgs)} benchmark images.")
    benchmark_data = []
    class_confidence_records = []

    # Warmup and timing: parameters num_warmup, num_runs passed to function

    for name, meta in models_info.items():
        print(f"\nBenchmarking {name}...")
        latencies = []
        scores_by_class = {}

        # Warmup
        for _ in range(num_warmup):
            for img in loaded_imgs:
                if meta['type'] == 'ultralytics':
                    _ = meta['model'].predict(source=img, conf=0.25, verbose=False)
                else:
                    tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        _ = meta['model']([tensor])

        # Timed Evaluation Runs
        for r in range(num_runs):
            for img in loaded_imgs:
                t0 = time.time()
                if meta['type'] == 'ultralytics':
                    res = meta['model'].predict(source=img, conf=0.25, verbose=False)[0]
                    dt = (time.time() - t0) * 1000.0
                    latencies.append(dt)
                    if res.boxes is not None:
                        for box in res.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            cname = meta['model'].names[cls_id]
                            scores_by_class.setdefault(cname, []).append(conf)
                else:
                    tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        preds = meta['model']([tensor])[0]
                    dt = (time.time() - t0) * 1000.0
                    latencies.append(dt)
                    for sc, lb in zip(preds['scores'].cpu().numpy(), preds['labels'].cpu().numpy()):
                        if sc >= 0.25:
                            cname = COCO_INSTANCE_CATEGORY_NAMES[lb] if lb < len(COCO_INSTANCE_CATEGORY_NAMES) else f"cls_{lb}"
                            if cname not in ['__background__', 'N/A']:
                                scores_by_class.setdefault(cname, []).append(float(sc))

        avg_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        fps = 1000.0 / avg_latency

        # Average confidence overall
        all_confs = [s for lst in scores_by_class.values() for s in lst]
        avg_conf = float(np.mean(all_confs)) if all_confs else 0.0

        for cname, slist in scores_by_class.items():
            class_confidence_records.append({
                'Model': name,
                'Class': cname,
                'Mean_Confidence': float(np.mean(slist)),
                'Detection_Count': len(slist)
            })

        print(f"  -> Mean Latency: {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
        print(f"  -> Speed: {fps:.2f} FPS")
        print(f"  -> Size: {meta['size_mb']:.2f} MB | Parameters: {meta['params']:,}")
        print(f"  -> Mean Detection Confidence: {avg_conf*100:.1f}%")

        benchmark_data.append({
            'Model': name,
            'Latency_ms': round(avg_latency, 2),
            'Latency_std_ms': round(std_latency, 2),
            'FPS': round(fps, 2),
            'Model_Size_MB': round(meta['size_mb'], 2),
            'Parameters': meta['params'],
            'Mean_Confidence': round(avg_conf, 4),
            'Total_Detections': len(all_confs)
        })

    # Save CSVs
    df_bench = pd.DataFrame(benchmark_data)
    csv_path = output_dir / "benchmark_metrics.csv"
    df_bench.to_csv(str(csv_path), index=False)
    print(f"\n[OK] Benchmark metrics saved to: {csv_path}")

    df_class = pd.DataFrame(class_confidence_records)
    class_csv_path = output_dir / "benchmark_class_confidences.csv"
    df_class.to_csv(str(class_csv_path), index=False)

    # Generate Publication Comparison Charts
    generate_charts(df_bench, df_class, output_dir / "benchmark_comparison_charts.png")

def generate_charts(df_bench, df_class, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.25, hspace=0.30)

    # 1. Inference Latency & FPS
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df_bench['Model'], df_bench['Latency_ms'], color=['#2ca02c', '#ff7f0e', '#1f77b4'], width=0.45)
    ax1.set_ylabel("Inference Time (ms / frame) [Lower is Better]", fontsize=11, fontweight='bold')
    ax1.set_title("Inference Latency per Frame (CPU)", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b, fps in zip(bars1, df_bench['FPS']):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + max(df_bench['Latency_ms'])*0.02,
                 f"{h:.1f} ms\n({fps:.1f} FPS)", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Model Size & Parameters
    ax2 = axes[0, 1]
    x = np.arange(len(df_bench))
    width = 0.35
    b_size = ax2.bar(x - width/2, df_bench['Model_Size_MB'], width, label='Model Disk Size (MB)', color='#9467bd')
    b_param = ax2.bar(x + width/2, [p / 1e6 for p in df_bench['Parameters']], width, label='Parameters (Millions)', color='#8c564b')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_bench['Model'])
    ax2.set_ylabel("Size (MB) / Parameters (M) [Lower is Better]", fontsize=11, fontweight='bold')
    ax2.set_title("Model Footprint & Computational Scale", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for b in b_size:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}MB", ha='center', fontsize=9)
    for b in b_param:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}M", ha='center', fontsize=9)

    # 3. Class-Wise Confidence Scores for Common Objects
    ax3 = axes[1, 0]
    top_classes = ['laptop', 'cup', 'keyboard', 'person', 'potted plant', 'chair']
    df_filtered = df_class[df_class['Class'].isin(top_classes)]
    if not df_filtered.empty:
        pivot = df_filtered.pivot_table(index='Class', columns='Model', values='Mean_Confidence').fillna(0)
        pivot.plot(kind='bar', ax=ax3, colormap='viridis', width=0.7)
        ax3.set_ylabel("Mean Confidence Score (0.0 - 1.0)", fontsize=11, fontweight='bold')
        ax3.set_title("Confidence Scores across Common Lab Asset Types", fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        ax3.legend(loc='lower right')
    else:
        ax3.text(0.5, 0.5, "Class confidence data collected", ha='center')

    # 4. Latency vs Footprint Trade-off
    ax4 = axes[1, 1]
    for _, row in df_bench.iterrows():
        ax4.scatter(row['Model_Size_MB'], row['Latency_ms'], s=row['Parameters']/5e4, 
                    alpha=0.7, edgecolors='black', linewidth=1.5, label=row['Model'])
        ax4.annotate(f"{row['Model']}\n({row['FPS']:.1f} FPS)", 
                     (row['Model_Size_MB'], row['Latency_ms']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
    ax4.set_xlabel("Model Disk Size (MB) [Lower is Better]", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Inference Latency (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    ax4.set_title("Operational Pareto Frontier (Speed vs. Size)", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Lab 09: Object Detection Comprehensive Benchmarks\nStudent: Muhammad John Raza (23k-0069) | Instructor: Talha Shahid", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved publication benchmark comparison charts: {save_path}")

if __name__ == "__main__":
    benchmark_models()
