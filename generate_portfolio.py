import os
import io
import sys
import base64
import contextlib
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf

class NotebookBuilder:
    def __init__(self, workdir):
        self.workdir = workdir
        self.nb = nbf.v4.new_notebook()
        self.nb.cells = []
        self.exec_count = 1
        self.globals_dict = {'__builtins__': __builtins__}

    def add_md(self, text):
        self.nb.cells.append(nbf.v4.new_markdown_cell(text))

    def add_code(self, code_str):
        cell = nbf.v4.new_code_cell(code_str)
        cell.execution_count = self.exec_count
        self.exec_count += 1
        
        # Execute code in working directory and capture output
        old_cwd = os.getcwd()
        os.chdir(self.workdir)
        stdout_buf = io.StringIO()
        figures = []

        def custom_show(*args, **kwargs):
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            figures.append(b64)
            plt.close(fig)

        plt.show = custom_show
        
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code_str, self.globals_dict)
        finally:
            os.chdir(old_cwd)
            
        outputs = []
        out_text = stdout_buf.getvalue()
        if out_text:
            outputs.append(nbf.v4.new_output('stream', name='stdout', text=out_text))
            
        for fig_b64 in figures:
            outputs.append(nbf.v4.new_output(
                'display_data',
                data={'image/png': fig_b64, 'text/plain': ['<Figure size 640x480 with 1 Axes>']},
                metadata={}
            ))
            
        cell.outputs = outputs
        self.nb.cells.append(cell)

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            nbf.write(self.nb, f)
        print(f'Successfully created and executed {filepath}')

def build_task_1():
    print('Processing Task 1...')
    nb = NotebookBuilder('Task_1_Chest_XRay')
    
    nb.add_md(
        "# Task 1: Diagnostic Enhancement of Chest X-Rays\n"
        "**Medical Imaging Portfolio** | Author: **23k0069**\n\n"
        "### Scenario\n"
        "Raw single-channel pulmonary X-rays frequently suffer from severe underexposure, masking structural details of the lung cavities and potential fluid accumulation.\n"
        "This pipeline applies rigorous mathematical transformations to restore dynamic contrast, apply false-color diagnostic mapping, neutralize color cast, threshold dense tissue, and apply logarithmic/gamma transformations."
    )
    
    nb.add_code(
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['image.cmap'] = 'gray'\n"
        "print('Environment and libraries initialized.')"
    )
    
    nb.add_md(
        "## 1. Load and Display Raw Grayscale Image\n"
        "Load the underexposed single-channel X-ray matrix from `data/sample_xray.png`."
    )
    
    nb.add_code(
        "raw = cv2.imread('data/sample_xray.png', cv2.IMREAD_GRAYSCALE)\n"
        "print(f'Dimensions: {raw.shape} | Intensity Min: {raw.min()}, Max: {raw.max()}')\n\n"
        "plt.figure(figsize=(6, 6))\n"
        "plt.imshow(raw, cmap='gray')\n"
        "plt.title('1. Raw Underexposed Chest X-Ray')\n"
        "plt.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/01_raw_xray.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 2. Contrast Enhancement (Histogram Equalization)\n"
        "Forcefully redistribute the intensity distribution across the dynamic range [0, 255] using cumulative distribution linearization."
    )
    
    nb.add_code(
        "eq = cv2.equalizeHist(raw)\n\n"
        "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
        "axes[0, 0].imshow(raw, cmap='gray')\n"
        "axes[0, 0].set_title('Raw Underexposed X-Ray')\n"
        "axes[0, 0].axis('off')\n\n"
        "axes[0, 1].imshow(eq, cmap='gray')\n"
        "axes[0, 1].set_title('Histogram Equalized Matrix')\n"
        "axes[0, 1].axis('off')\n\n"
        "axes[1, 0].hist(raw.ravel(), bins=256, range=[0, 256], color='#2b5c8f')\n"
        "axes[1, 0].set_title('Raw Intensity Histogram')\n"
        "axes[1, 0].set_xlim([0, 256])\n\n"
        "axes[1, 1].hist(eq.ravel(), bins=256, range=[0, 256], color='#d95f02')\n"
        "axes[1, 1].set_title('Equalized Intensity Histogram')\n"
        "axes[1, 1].set_xlim([0, 256])\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/02_histogram_equalization.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 3. False-Color Mapping (COLORMAP_JET)\n"
        "Convert the equalized grayscale matrix into a pseudocolor heatmap using `COLORMAP_JET` to accentuate fluid-tissue density gradients."
    )
    
    nb.add_code(
        "heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
        "axes[0].imshow(eq, cmap='gray')\n"
        "axes[0].set_title('Equalized Grayscale')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('False-Color Heatmap (Fluid Boundary Detection)')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/03_colormap_jet.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 4. Color Balance (Lighting Correction)\n"
        "Simulate lighting correction by applying a Gray-World mathematical color balance shift to neutralize artificial color bias."
    )
    
    nb.add_code(
        "b, g, r = cv2.split(heat)\n"
        "mb, mg, mr = b.mean(), g.mean(), r.mean()\n"
        "avg = (mb + mg + mr) / 3.0\n\n"
        "b_cb = np.clip(b * (avg / mb), 0, 255).astype(np.uint8)\n"
        "g_cb = np.clip(g * (avg / mg), 0, 255).astype(np.uint8)\n"
        "r_cb = np.clip(r * (avg / mr), 0, 255).astype(np.uint8)\n"
        "cb = cv2.merge([b_cb, g_cb, r_cb])\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
        "axes[0].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))\n"
        "axes[0].set_title('Unbalanced Heatmap')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('Gray-World Color Balanced')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/04_color_balance.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 5. Color Filtering (Dense Tissue Thresholding)\n"
        "Dense tissues (cortical bone, clavicles, ribs, infiltrates) reflect high intensities. Apply strict mathematical thresholding to segment them."
    )
    
    nb.add_code(
        "_, mask = cv2.threshold(eq, 180, 255, cv2.THRESH_BINARY)\n"
        "dense = cv2.bitwise_and(cb, cb, mask=mask)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "axes[0].imshow(eq, cmap='gray')\n"
        "axes[0].set_title('Equalized Matrix')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(mask, cmap='gray')\n"
        "axes[1].set_title('Dense Tissue Binary Mask')\n"
        "axes[1].axis('off')\n\n"
        "axes[2].imshow(cv2.cvtColor(dense, cv2.COLOR_BGR2RGB))\n"
        "axes[2].set_title('Segmented Dense Structures')\n"
        "axes[2].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/05_dense_tissue_mask.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 6. Logarithmic Transformation\n"
        "Expand compressed low-intensity values using $s = c \\cdot \\ln(1 + r)$ to unveil peripheral ribcage boundaries."
    )
    
    nb.add_code(
        "c_log = 255.0 / np.log(1.0 + np.max(raw))\n"
        "log_img = (c_log * np.log(1.0 + raw.astype(np.float32))).astype(np.uint8)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
        "axes[0].imshow(raw, cmap='gray')\n"
        "axes[0].set_title('Raw Underexposed X-Ray')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(log_img, cmap='gray')\n"
        "axes[1].set_title('Logarithmic Expansion (Ribcage Revealed)')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/06_log_transformed.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 7. Power-Law (Gamma) Transformation\n"
        "Apply fractional power-law $s = 255 \\cdot (r / 255)^\\gamma$ with $\\gamma < 1.0$ ($\\gamma = 0.55$) to lift midtone soft-tissue contrast while preventing bone over-saturation."
    )
    
    nb.add_code(
        "gamma = 0.55\n"
        "gam_img = np.clip(255.0 * ((raw / 255.0) ** gamma), 0, 255).astype(np.uint8)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "axes[0].imshow(raw, cmap='gray')\n"
        "axes[0].set_title('Raw X-Ray')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(log_img, cmap='gray')\n"
        "axes[1].set_title('Logarithmic Transform')\n"
        "axes[1].axis('off')\n\n"
        "axes[2].imshow(gam_img, cmap='gray')\n"
        "axes[2].set_title(f'Gamma Corrected (γ = {gamma})')\n"
        "axes[2].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/07_gamma_transformed.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 8. Complete Pipeline Diagnostic Overview"
    )
    
    nb.add_code(
        "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n"
        "axes[0, 0].imshow(raw, cmap='gray')\n"
        "axes[0, 0].set_title('1. Raw Underexposed')\n"
        "axes[0, 0].axis('off')\n\n"
        "axes[0, 1].imshow(eq, cmap='gray')\n"
        "axes[0, 1].set_title('2. Equalized')\n"
        "axes[0, 1].axis('off')\n\n"
        "axes[0, 2].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))\n"
        "axes[0, 2].set_title('3. Jet Colormap')\n"
        "axes[0, 2].axis('off')\n\n"
        "axes[1, 0].imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))\n"
        "axes[1, 0].set_title('4. Color Balanced')\n"
        "axes[1, 0].axis('off')\n\n"
        "axes[1, 1].imshow(cv2.cvtColor(dense, cv2.COLOR_BGR2RGB))\n"
        "axes[1, 1].set_title('5. Dense Tissue Mask')\n"
        "axes[1, 1].axis('off')\n\n"
        "axes[1, 2].imshow(gam_img, cmap='gray')\n"
        "axes[1, 2].set_title(f'6. Gamma Enhanced (γ={gamma})')\n"
        "axes[1, 2].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/08_pipeline_summary.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.save('Task_1_Chest_XRay/xray_enhancement.ipynb')

def build_task_2():
    print('Processing Task 2...')
    nb = NotebookBuilder('Task_2_Cardiac_Fusion')
    
    nb.add_md(
        "# Task 2: Multi-Modal Cardiac Image Fusion\n"
        "**Medical Imaging Portfolio** | Author: **23k0069**\n\n"
        "### Scenario\n"
        "Diagnosing cardiac disease requires simultaneous evaluation of hard anatomical boundaries (captured by CT) and subtle myocardial soft-tissue variations (captured by MRI).\n"
        "This pipeline performs multi-modal matrix fusion: normalizing dynamic ranges, mapping distinct pseudocolor spectra, applying weighted blending ($\alpha > \beta$), and safeguarding against saturation using logarithmic and gamma curves."
    )
    
    nb.add_code(
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['image.cmap'] = 'gray'\n"
        "print('Fusion environment initialized.')"
    )
    
    nb.add_md(
        "## 1. Load Modalities\n"
        "Load aligned cross-sectional slices: CT (`data/ct_slice.png`) and MRI (`data/mri_slice.png`)."
    )
    
    nb.add_code(
        "ct = cv2.imread('data/ct_slice.png', cv2.IMREAD_GRAYSCALE)\n"
        "mri = cv2.imread('data/mri_slice.png', cv2.IMREAD_GRAYSCALE)\n"
        "print(f'CT Matrix: {ct.shape} | MRI Matrix: {mri.shape}')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n"
        "axes[0].imshow(ct, cmap='gray')\n"
        "axes[0].set_title('Raw CT Slice (Bone / Structural Edges)')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(mri, cmap='gray')\n"
        "axes[1].set_title('Raw MRI Slice (Soft Tissue / Myocardium)')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/01_raw_modalities.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 2. Independent Histogram Equalization\n"
        "Equalize each modality separately to maximize individual dynamic range prior to combination."
    )
    
    nb.add_code(
        "ct_eq = cv2.equalizeHist(ct)\n"
        "mri_eq = cv2.equalizeHist(mri)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n"
        "axes[0].imshow(ct_eq, cmap='gray')\n"
        "axes[0].set_title('Equalized CT Matrix')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(mri_eq, cmap='gray')\n"
        "axes[1].set_title('Equalized MRI Matrix')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/02_equalized_modalities.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 3. Color Mapping and Visual Differentiation\n"
        "Convert CT to `COLORMAP_BONE` (sharp anatomical boundaries) and MRI to `COLORMAP_JET` (subtle myocardial gradient transitions)."
    )
    
    nb.add_code(
        "ct_col = cv2.applyColorMap(ct_eq, cv2.COLORMAP_BONE)\n"
        "mri_col = cv2.applyColorMap(mri_eq, cv2.COLORMAP_JET)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n"
        "axes[0].imshow(cv2.cvtColor(ct_col, cv2.COLOR_BGR2RGB))\n"
        "axes[0].set_title('CT Pseudocolor (COLORMAP_BONE)')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(mri_col, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('MRI Pseudocolor (COLORMAP_JET)')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/03_modality_colormaps.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 4. Multi-Modal Weighted Fusion\n"
        "Blend matrices via linear weighted superposition using `cv2.addWeighted()`:\n"
        "$$\\text{Fused} = \\alpha \\cdot \\text{CT} + \\beta \\cdot \\text{MRI} + 0$$\n"
        "Assign higher weight to CT ($\\alpha = 0.65$) for edge integrity and MRI ($\\beta = 0.35$) for soft-tissue variations."
    )
    
    nb.add_code(
        "alpha = 0.65\n"
        "beta = 0.35\n"
        "fused = cv2.addWeighted(ct_col, alpha, mri_col, beta, 0)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "axes[0].imshow(cv2.cvtColor(ct_col, cv2.COLOR_BGR2RGB))\n"
        "axes[0].set_title('Enhanced CT')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(mri_col, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('Enhanced MRI')\n"
        "axes[1].axis('off')\n\n"
        "axes[2].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))\n"
        "axes[2].set_title(f'Weighted Blending (α={alpha}, β={beta})')\n"
        "axes[2].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/04_weighted_fusion.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 5. Logarithmic and Power-Law Dynamic Range Preservation\n"
        "Safeguard against clipped highlights and crushed shadows resulting from matrix addition through logarithmic re-scaling and gamma tuning ($\\gamma = 0.85$)."
    )
    
    nb.add_code(
        "c_log = 255.0 / np.log(1.0 + np.max(fused))\n"
        "fused_log = (c_log * np.log(1.0 + fused.astype(np.float32))).astype(np.uint8)\n\n"
        "gamma = 0.85\n"
        "fused_final = np.clip(255.0 * ((fused_log / 255.0) ** gamma), 0, 255).astype(np.uint8)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
        "axes[0].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))\n"
        "axes[0].set_title('Linear Fused Matrix')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(fused_final, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('Corrected Output (Log + Power-Law)')\n"
        "axes[1].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/05_fused_enhanced.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.add_md(
        "## 6. Comparative Diagnostic Analysis\n"
        "Side-by-side comparative inspection validating edge preservation and soft-tissue visibility."
    )
    
    nb.add_code(
        "comp = np.hstack([ct_col, mri_col, fused_final])\n"
        "cv2.imwrite('output/06_comparative_fusion_banner.png', comp)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 6))\n"
        "axes[0].imshow(cv2.cvtColor(ct_col, cv2.COLOR_BGR2RGB))\n"
        "axes[0].set_title('1. Standalone CT (Hard Structures)')\n"
        "axes[0].axis('off')\n\n"
        "axes[1].imshow(cv2.cvtColor(mri_col, cv2.COLOR_BGR2RGB))\n"
        "axes[1].set_title('2. Standalone MRI (Soft Tissue)')\n"
        "axes[1].axis('off')\n\n"
        "axes[2].imshow(cv2.cvtColor(fused_final, cv2.COLOR_BGR2RGB))\n"
        "axes[2].set_title('3. Multi-Modal Fused Diagnostic View')\n"
        "axes[2].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('output/06_comparative_analysis.png', bbox_inches='tight')\n"
        "plt.show()"
    )
    
    nb.save('Task_2_Cardiac_Fusion/modal_fusion.ipynb')

def build_task_3():
    print('Processing Task 3...')
    nb = NotebookBuilder('Task_3_Echo_Analysis')
    
    nb.add_md(
        "# Task 3: Real-Time Echocardiogram Video Analysis\n"
        "**Medical Imaging Portfolio** | Author: **23k0069**\n\n"
        "### Scenario\n"
        "Ultrasound echocardiogram footage is notoriously murky, low-contrast, and degraded by probe acoustic backscatter.\n"
        "This real-time pipeline captures video frame-by-frame, equalizing contrast, mapping pseudocolor jet flow, balancing chrominance, expanding dark chambers via logarithmic curves, and attenuating backscatter using power-law tuning."
    )
    
    nb.add_code(
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "plt.rcParams['figure.figsize'] = (12, 6)\n"
        "print('Video analysis pipeline ready.')"
    )
    
    nb.add_md(
        "## 1. Video Capture Setup\n"
        "Initialize `cv2.VideoCapture` from `data/echocardiogram.mp4` and inspect stream dimensions."
    )
    
    nb.add_code(
        "cap = cv2.VideoCapture('data/echocardiogram.mp4')\n"
        "fps = cap.get(cv2.CAP_PROP_FPS)\n"
        "count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n"
        "h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
        "cap.release()\n\n"
        "print(f'Echo Stream: {w}x{h} Resolution | {fps} FPS | Total Frame Count: {count}')"
    )
    
    nb.add_md(
        "## 2. Frame Mathematical Enhancement Function\n"
        "Matrix processing per frame:\n"
        "1. Single-channel luminance conversion\n"
        "2. Histogram Equalization (combats murky acoustic attenuation)\n"
        "3. False-Color Mapping (`COLORMAP_JET` highlighting blood flow velocities)\n"
        "4. Gray-World Color Balance\n"
        "5. Logarithmic Transformation (reveals dark ventricular recesses)\n"
        "6. Power-Law Suppression ($\\gamma = 1.35$ suppresses blinding white probe backscatter)"
    )
    
    nb.add_code(
        "def enhance_frame(frame):\n"
        "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame\n"
        "    \n"
        "    # 1. Histogram Equalization\n"
        "    eq = cv2.equalizeHist(gray)\n"
        "    \n"
        "    # 2. Color Mapping (Jet)\n"
        "    heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)\n"
        "    \n"
        "    # 3. Color Balance (Gray-World)\n"
        "    b, g, r = cv2.split(heat)\n"
        "    mb, mg, mr = b.mean(), g.mean(), r.mean()\n"
        "    avg = (mb + mg + mr) / 3.0\n"
        "    b_cb = np.clip(b * (avg / mb), 0, 255).astype(np.uint8)\n"
        "    g_cb = np.clip(g * (avg / mg), 0, 255).astype(np.uint8)\n"
        "    r_cb = np.clip(r * (avg / mr), 0, 255).astype(np.uint8)\n"
        "    cb = cv2.merge([b_cb, g_cb, r_cb])\n"
        "    \n"
        "    # 4. Logarithmic Transformation\n"
        "    c_log = 255.0 / np.log(1.0 + np.max(cb))\n"
        "    log_f = (c_log * np.log(1.0 + cb.astype(np.float32))).astype(np.uint8)\n"
        "    \n"
        "    # 5. Power-Law (Gamma = 1.35 suppresses white backscatter)\n"
        "    gamma = 1.35\n"
        "    enh = np.clip(255.0 * ((log_f / 255.0) ** gamma), 0, 255).astype(np.uint8)\n"
        "    return enh\n\n"
        "print('Enhancement function compiled.')"
    )
    
    nb.add_md(
        "## 3. Real-Time Video Processing Loop\n"
        "Iterate through the video stream, process each frame, and save the side-by-side concatenated monitoring array to `output/side_by_side_echo.mp4`."
    )
    
    nb.add_code(
        "cap = cv2.VideoCapture('data/echocardiogram.mp4')\n"
        "out = cv2.VideoWriter('output/side_by_side_echo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))\n\n"
        "sample_frames = []\n"
        "idx = 0\n\n"
        "while cap.isOpened():\n"
        "    ret, frame = cap.read()\n"
        "    if not ret:\n"
        "        break\n"
        "    \n"
        "    enh = enhance_frame(frame)\n"
        "    side_by_side = np.hstack([frame, enh])\n"
        "    out.write(side_by_side)\n"
        "    \n"
        "    if idx in [10, 25, 40]:\n"
        "        sample_frames.append((idx, frame, enh))\n"
        "    idx += 1\n\n"
        "cap.release()\n"
        "out.release()\n"
        "print(f'Processed {idx} frames. Video saved to output/side_by_side_echo.mp4')"
    )
    
    nb.add_md(
        "## 4. Visual Inspection of Monitored Ultrasound Frames\n"
        "Side-by-side comparison across key stages of ventricular systole and diastole."
    )
    
    nb.add_code(
        "for frame_idx, raw_f, enh_f in sample_frames:\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "    axes[0].imshow(cv2.cvtColor(raw_f, cv2.COLOR_BGR2RGB))\n"
        "    axes[0].set_title(f'Raw Ultrasound Stream (Frame {frame_idx})')\n"
        "    axes[0].axis('off')\n\n"
        "    axes[1].imshow(cv2.cvtColor(enh_f, cv2.COLOR_BGR2RGB))\n"
        "    axes[1].set_title(f'Enhanced Real-Time Stream (Frame {frame_idx})')\n"
        "    axes[1].axis('off')\n\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(f'output/echo_frame_{frame_idx}_comparison.png', bbox_inches='tight')\n"
        "    plt.show()\n\n"
        "rep_frame = np.hstack([sample_frames[1][1], sample_frames[1][2]])\n"
        "cv2.imwrite('output/side_by_side_frame.png', rep_frame)\n"
        "print('Saved representative frame to output/side_by_side_frame.png')"
    )
    
    nb.save('Task_3_Echo_Analysis/realtime_echo.ipynb')

if __name__ == '__main__':
    build_task_1()
    build_task_2()
    build_task_3()
    print('All portfolio tasks completed successfully!')
