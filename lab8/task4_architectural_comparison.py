"""
Task 4: Architectural Comparison Summary (CNN vs. R-CNN vs. Fast R-CNN vs. Faster R-CNN)
========================================================================================
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 08)

Matches sections 2.1, 2.2, 2.3, 2.4 of the Lab Manual:
- In-depth architectural, mathematical, and computational analysis.
- Detailed comparative table.
- Generates high-resolution visualization summary figure.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

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
os.makedirs(output_dir, exist_ok=True)

def generate_architectural_comparison():
    print("=================================================================")
    print("      LAB 08 - TASK 4: ARCHITECTURAL COMPARISON SUMMARY          ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")

    comparison_data = [
        {
            "Algorithm": "Standard CNN (Sliding Window)",
            "Region Proposal Method": "Dense Sliding Windows (multi-scale, multi-aspect)",
            "Feature Extraction": "Separate CNN forward pass for every sliding window",
            "RoI Processing": "Crop and scale each window before CNN",
            "Prediction Time / Image": "Hundreds of seconds / Extremely slow",
            "Training Paradigm": "Classification only, no bounding box regression",
            "Key Limitations": "Massive computational redundancy; billions of candidate crops."
        },
        {
            "Algorithm": "R-CNN (Girshick et al., 2014)",
            "Region Proposal Method": "Selective Search (~2,000 candidate proposals)",
            "Feature Extraction": "Separate CNN forward pass for each warped proposal (~2,000 passes)",
            "RoI Processing": "Affine warping to fixed size (e.g. 224x224)",
            "Prediction Time / Image": "40 - 50 seconds",
            "Training Paradigm": "Multi-stage: 1) ConvNet pretraining, 2) SVM training, 3) BBox regressor",
            "Key Limitations": "Huge disk storage for features; slow inference; non-differentiable proposal generation."
        },
        {
            "Algorithm": "Fast R-CNN (Girshick, 2015)",
            "Region Proposal Method": "Selective Search (External CPU algorithm)",
            "Feature Extraction": "Single CNN forward pass on the entire image to obtain feature map",
            "RoI Processing": "RoI Pooling layer directly on the shared conv5 feature map",
            "Prediction Time / Image": "2.0 seconds",
            "Training Paradigm": "Single-stage multi-task loss (Softmax + Smooth L1 regression)",
            "Key Limitations": "Selective search remains the computational bottleneck (takes ~2s on CPU)."
        },
        {
            "Algorithm": "Faster R-CNN (Ren et al., 2015)",
            "Region Proposal Method": "Region Proposal Network (RPN) sharing conv layers with detector",
            "Feature Extraction": "Single CNN forward pass; features shared by both RPN and Fast R-CNN head",
            "RoI Processing": "RoI Pooling / RoIAlign from RPN proposals",
            "Prediction Time / Image": "0.2 seconds (5 - 10 FPS, Real-time capable)",
            "Training Paradigm": "Fully end-to-end differentiable 4-loss joint training",
            "Key Limitations": "Higher structural complexity; two-stage sequential dependency."
        }
    ]

    df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 110)
    print("                         R-CNN FAMILY ARCHITECTURAL COMPARISON SUMMARY                         ")
    print("=" * 110)
    for row in comparison_data:
        print(f"\n--- {row['Algorithm']} ---")
        print(f"  * Proposal Method:     {row['Region Proposal Method']}")
        print(f"  * Feature Extraction:  {row['Feature Extraction']}")
        print(f"  * RoI Processing:      {row['RoI Processing']}")
        print(f"  * Inference Speed:     {row['Prediction Time / Image']}")
        print(f"  * Training Paradigm:   {row['Training Paradigm']}")
        print(f"  * Limitations:         {row['Key Limitations']}")
    print("=" * 110)

    # Save to text report
    txt_path = os.path.join(output_dir, "architectural_comparison_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df.to_string())

    # Create visualization graphic
    fig = plt.figure(figsize=(18, 10))

    # Bar chart comparing inference latency (log scale)
    ax1 = fig.add_subplot(1, 2, 1)
    models_list = ["Standard CNN", "R-CNN", "Fast R-CNN", "Faster R-CNN"]
    times = [150.0, 45.0, 2.0, 0.2]
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']

    bars = ax1.bar(models_list, times, color=colors, edgecolor='black', width=0.55)
    ax1.set_yscale('log')
    ax1.set_ylabel('Inference Time per Image (Seconds, Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Speed Comparison (Lower is Faster)', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    for bar, t in zip(bars, times):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{t}s", ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    # Evolution flowchart & summary table
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

    table_cells = [
        ["Model", "Proposal Method", "Shared Features", "Inference Time", "FPS"],
        ["CNN", "Sliding Window", "No (Per Window)", "~150s", "<0.01"],
        ["R-CNN", "Selective Search", "No (~2k passes)", "40-50s", "0.02"],
        ["Fast R-CNN", "Selective Search", "Yes (RoI Pool)", "2s", "0.5"],
        ["Faster R-CNN", "RPN (Learned)", "Yes (Full End-to-End)", "0.2s", "5 - 10"]
    ]

    table = ax2.table(cellText=table_cells, loc='center', cellLoc='center', colWidths=[0.22, 0.28, 0.25, 0.15, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 2.6)

    # Style header row
    for col_idx in range(5):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#2b3e50')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

    # Colorize model rows
    row_colors = ['#fdf2f2', '#fef8f0', '#f0f9fd', '#f0fbf0']
    for row_idx in range(1, 5):
        for col_idx in range(5):
            table[(row_idx, col_idx)].set_facecolor(row_colors[row_idx - 1])

    ax2.set_title("Architectural Evolution & Key Milestones", fontsize=13, fontweight='bold', pad=20)

    plt.suptitle("Lab 08: R-CNN, Fast R-CNN & Faster R-CNN Architectural Analysis\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "architectural_comparison_summary.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[*] Saved comparison chart to {chart_path}")
    return chart_path

if __name__ == "__main__":
    generate_architectural_comparison()
