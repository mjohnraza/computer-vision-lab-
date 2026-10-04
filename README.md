# Medical Imaging Portfolio — Lab 02

**Author:** 23k0069  
**Course:** Medical Imaging & Computer Vision  

---

## Repository Architecture

```text
StudentName_Medical_Imaging_Portfolio/
├── Task_1_Chest_XRay/
│   ├── data/                   # The specific X-ray images used for testing
│   ├── output/                 # Screenshots/saved images of your enhanced results
│   ├── xray_enhancement.ipynb  # Jupyter notebook with complete execution outputs
│   └── README.md               # Explanation of pulmonary enhancement pipeline
│
├── Task_2_Cardiac_Fusion/
│   ├── data/                   # The specific CT and MRI matched pairs used
│   ├── output/                 # Saved fused heatmaps and comparison charts
│   ├── modal_fusion.ipynb      # Jupyter notebook with complete execution outputs
│   └── README.md               # Fusion weighting logic and comparative analysis
│
├── Task_3_Echo_Analysis/
│   ├── data/                   # The short ultrasound .mp4 snippet used
│   ├── output/                 # Screenshots of the side-by-side video pipeline
│   ├── realtime_echo.ipynb     # Jupyter notebook with complete execution outputs
│   ├── run_echo_live.py        # Real-time desktop live video display loop
│   └── README.md               # Instructions on how to run video loop
│
├── requirements.txt            # Global library dependencies (OpenCV, NumPy, etc.)
└── README.md                   # Main repository overview and setup instructions
```

---

## Dataset Handling (The 100MB Rule Compliance)

In accordance with laboratory submission constraints:
- All datasets have been trimmed and curated to specific, lightweight sample files strictly adhering to GitHub's file size limit (< 100 MB).
- **Task 1 Data**: `sample_xray.png` (512×512 single-channel underexposed radiograph).
- **Task 2 Data**: Registered cardiac slices `ct_slice.png` and `mri_slice.png` (256×256 matched cross-sections).
- **Task 3 Data**: `echocardiogram.mp4` (~230 KB, 50-frame clinical echocardiogram ultrasound clip).

All Python scripts and notebooks strictly use **relative paths** (e.g., `data/sample_xray.png`), ensuring immediate portability across any operating system without hardcoded absolute paths.

---

## Environment Setup & Installation

Clone or extract the repository, navigate into the root directory, and install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Tasks Summary

### Task 1: Diagnostic Enhancement of Chest X-Rays
- **Goal**: Restore structural clarity to severely underexposed chest radiographs.
- **Operations**:
  1. Grayscale loading and intensity profiling.
  2. Forceful contrast enhancement via **Histogram Equalization** (`cv2.equalizeHist`).
  3. False-color mapping via `cv2.COLORMAP_JET` to highlight subtle fluid-tissue interfaces.
  4. Mathematical **Gray-World color balance** to eliminate artificial color casts.
  5. Dense tissue thresholding to isolate bone structures and dense consolidation.
  6. **Logarithmic transformation** ($s = c \cdot \ln(1 + r)$) to expand compressed low-intensity lung perimeters.
  7. Fractional **Power-Law (Gamma) transformation** ($\gamma = 0.55 < 1.0$) to lift midtone contrast while suppressing harsh bone glare.

### Task 2: Multi-Modal Cardiac Image Fusion
- **Goal**: Unify complementary diagnostic properties of CT (hard structural edges) and MRI (soft myocardial tissue).
- **Operations**:
  1. Modality loading: CT and MRI aligned slices.
  2. Independent histogram equalization to maximize individual dynamic ranges.
  3. Distinct pseudocolor encoding (`COLORMAP_BONE` for CT, `COLORMAP_JET` for MRI).
  4. **Weighted linear superposition** via `cv2.addWeighted()` with $\alpha = 0.65$ (CT structural emphasis) and $\beta = 0.35$ (MRI tissue variation).
  5. Logarithmic and power-law ($\gamma = 0.85$) stabilization to prevent saturation clipping.
  6. Side-by-side comparative clinical evaluation.

### Task 3: Real-Time Echocardiogram Video Analysis
- **Goal**: Dynamic frame-by-frame ultrasound stream enhancement.
- **Operations**:
  1. Capture initialization from `data/echocardiogram.mp4`.
  2. Frame-by-frame real-time enhancement loop:
     - Histogram equalization combating acoustic attenuation.
     - `COLORMAP_JET` pseudocolor mapping indicating flow dynamics.
     - Gray-World channel balancing.
     - Logarithmic dark chamber cavity expansion.
     - Power-law suppression ($\gamma = 1.35$) attenuating blinding transducer backscatter noise.
  3. Real-time horizontal concatenation (`[raw_feed, enhanced_feed]`) written to `output/side_by_side_echo.mp4`.
  4. Live desktop GUI player available via `python run_echo_live.py` (press `Q` to quit).

---

## Verification & Execution

To inspect the notebooks with pre-rendered graphical and analytical outputs, simply open:
- `Task_1_Chest_XRay/xray_enhancement.ipynb`
- `Task_2_Cardiac_Fusion/modal_fusion.ipynb`
- `Task_3_Echo_Analysis/realtime_echo.ipynb`

To re-run and re-generate all outputs and notebooks programmatically:
```bash
python generate_portfolio.py
```
