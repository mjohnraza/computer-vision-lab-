# Task 3: Real-Time Echocardiogram Video Analysis

## Overview
This module implements a real-time OpenCV stream processing pipeline for cardiac ultrasound (echocardiogram) video. Echocardiograms suffer from acoustic speckle noise, low contrast, and probe near-field backscatter. This pipeline cleanses each frame and displays a live side-by-side feed comparing the raw stream with the mathematically enhanced diagnostics.

## Pipeline Architecture & Operations

For every video frame extracted in real-time from `data/echocardiogram.mp4`:
1. **Luminance Extraction**: Converts BGR frames to single-channel grayscale.
2. **Histogram Equalization**: Forcefully redistributes ultrasound intensities via `cv2.equalizeHist()`, eliminating murky acoustic attenuation.
3. **Color Mapping (`COLORMAP_JET`)**: Maps grayscale intensities to pseudocolor heatmaps, making blood flow dynamics and wall velocities immediately distinct.
4. **Gray-World Color Balance**: Corrects chromatic channel shifts across blue, green, and red channels.
5. **Logarithmic Expansion**: Applies $s = c \cdot \ln(1 + r)$ to expand the darkest recesses of the ventricles and atria.
6. **Power-Law Suppression**: Applies $\gamma = 1.35 > 1.0$ ($s = 255 \cdot (r/255)^\gamma$) to suppress the blinding white backscatter acoustic glare typical of ultrasound transducer probes.
7. **Monitoring Array**: Concatenates raw frame and enhanced frame horizontally (`np.hstack([frame, enh])`).

## Running the Pipeline

### Option 1: Jupyter Notebook
Open `realtime_echo.ipynb` to view the notebook with pre-executed inline figures across key cardiac phases and inspect the generated output video.

### Option 2: Live Desktop Window (cv2.imshow)
To watch the real-time pipeline render live in an interactive desktop window:
```bash
cd Task_3_Echo_Analysis
python run_echo_live.py
```
- Press **`Q`** at any time to exit the live display window.

## Outputs
All generated outputs are saved to `Task_3_Echo_Analysis/output/`:
- `side_by_side_echo.mp4`: Full rendered 50-frame side-by-side video stream.
- `side_by_side_frame.png`: High-resolution representative comparison frame.
- `echo_frame_10_comparison.png`: Cardiac contraction comparison (Frame 10).
- `echo_frame_25_comparison.png`: Mid-systolic comparison (Frame 25).
- `echo_frame_40_comparison.png`: End-diastolic comparison (Frame 40).
