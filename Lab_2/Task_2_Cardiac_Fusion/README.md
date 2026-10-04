# Task 2: Multi-Modal Cardiac Image Fusion

## Overview
This module mathematically fuses aligned Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) cardiac slices. CT offers high spatial resolution for dense bone and calcified structures, whereas MRI provides superior soft-tissue contrast for myocardial perfusion and chamber geometry.

## Fusion Weighting Logic & Pipeline

1. **Independent Pre-Processing**:
   - Both CT (`data/ct_slice.png`) and MRI (`data/mri_slice.png`) are loaded as single-channel matrices.
   - Each modality is independently equalized using `cv2.equalizeHist()` to maximize dynamic range before fusion.

2. **Modality Pseudocolor Assignment**:
   - **CT**: Converted using `cv2.COLORMAP_BONE` to accentuate bone boundaries and vascular borders.
   - **MRI**: Converted using `cv2.COLORMAP_JET` to accentuate myocardial tissue density gradients.

3. **Multi-Modal Weighted Fusion**:
   - Fused via linear weighted matrix addition (`cv2.addWeighted`):
     $$\text{Fused}(x, y) = \alpha \cdot \text{CT}(x, y) + \beta \cdot \text{MRI}(x, y) + 0$$
   - **Weight Allocation**:
     - $\alpha = 0.65$: Heavier weight given to CT to retain sharp anatomical boundaries and structural landmarks.
     - $\beta = 0.35$: Lighter weight given to MRI to overlay soft tissue and myocardial density variations without obscuring CT edges.

4. **Dynamic Range Preservation (Log + Power-Law)**:
   - Superposition of matrices can push high intensities toward saturation ($> 255$) or compress dark gradients.
   - A logarithmic transformation ($s = c \cdot \ln(1 + r)$) expands compressed lower values.
   - A power-law curve ($\gamma = 0.85$) balances midtone contrast, preventing blown-out highlights.

5. **Comparative Analysis**:
   - The standalone CT frame, standalone MRI frame, and final fused output are compared side-by-side to visually and mathematically validate registration and edge preservation.

## Outputs
All generated outputs are saved to `Task_2_Cardiac_Fusion/output/`:
- `01_raw_modalities.png`: Raw CT and MRI inputs.
- `02_equalized_modalities.png`: Independently equalized inputs.
- `03_modality_colormaps.png`: Pseudocolor representations.
- `04_weighted_fusion.png`: Direct weighted blend.
- `05_fused_enhanced.png`: Log and gamma corrected fusion.
- `06_comparative_analysis.png`: Side-by-side clinical comparison panel.
- `06_comparative_fusion_banner.png`: Concatenated banner matrix.
