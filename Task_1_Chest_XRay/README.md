# Task 1: Diagnostic Enhancement of Chest X-Rays

## Overview
This module enhances underexposed single-channel pulmonary X-rays to maximize thoracic structure visibility, fluid cavity delineation, and dense anatomical tissue isolation.

## Pipeline Architecture & Mathematical Operations

1. **Load and Display**:
   - Reads the single-channel matrix `data/sample_xray.png` using relative pathing:
     `cv2.imread('data/sample_xray.png', cv2.IMREAD_GRAYSCALE)`.
   - Inspects matrix dimensions and pixel intensity distribution ($[0, 255]$).

2. **Contrast Enhancement (Histogram Equalization)**:
   - Equalizes the intensity histogram via `cv2.equalizeHist()`.
   - Linearizes cumulative probability density to uncover obscured pulmonary textures and alveolar structures.

3. **Color Mapping (COLORMAP_JET)**:
   - Maps the equalized scalar intensity field to a 3-channel pseudocolor space via `cv2.applyColorMap(eq, cv2.COLORMAP_JET)`.
   - Human visual perception has significantly greater chromatic sensitivity than grayscale sensitivity, allowing clinicians to distinguish subtle fluid boundaries and pneumonia infiltrates.

4. **Color Balance (Gray-World Lighting Correction)**:
   - Normalizes channel chrominance using the Gray-World assumption:
     $$I_c' = \min\left(255, I_c \cdot \frac{\mu_{\text{gray}}}{\mu_c}\right), \quad c \in \{B, G, R\}$$
   - Eliminates artificial chromatic saturation and visual tint.

5. **Color Filtering (Dense Tissue Thresholding)**:
   - Segments high-density structures (ribs, clavicle, spine, and dense consolidation) via binary thresholding:
     $$M(x, y) = \begin{cases} 255 & \text{if } I_{\text{eq}}(x, y) \ge 180 \\ 0 & \text{otherwise} \end{cases}$$
   - Isolates dense anatomy using bitwise masking (`cv2.bitwise_and`).

6. **Logarithmic Transformation**:
   - Expands suppressed low-intensity dark regions while compressing highlights:
     $$s = c \cdot \ln(1 + r), \quad c = \frac{255}{\ln(1 + \max(r))}$$
   - Exposes faint ribcage boundaries and pleural contours.

7. **Power-Law (Gamma) Transformation**:
   - Applies fractional gamma adjustment ($\gamma = 0.55 < 1.0$):
     $$s = 255 \cdot \left(\frac{r}{255}\right)^\gamma$$
   - Enhances soft parenchymal midtones without over-saturating cortical bone.

## Outputs
All generated outputs are saved to `Task_1_Chest_XRay/output/`:
- `01_raw_xray.png`: Raw underexposed baseline image.
- `02_histogram_equalization.png`: Side-by-side comparison and histogram distribution.
- `03_colormap_jet.png`: Jet false-color heatmap.
- `04_color_balance.png`: Balanced visualization.
- `05_dense_tissue_mask.png`: Binary mask and segmented dense tissues.
- `06_log_transformed.png`: Logarithmic low-end dynamic expansion.
- `07_gamma_transformed.png`: Gamma-adjusted midtone enhancement.
- `08_pipeline_summary.png`: Unified 6-stage clinical comparison matrix.
