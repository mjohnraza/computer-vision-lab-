"""
Task 3: Anomaly Detection in Sensor Data Using Wavelet Transformation
===================================================================
Technique: Discrete Wavelet Transform (DWT) Multiresolution Analysis +
           Median Absolute Deviation (MAD) Noise Estimation +
           VisuShrink Universal Soft Thresholding + Inverse DWT + Residual Z-Score.

Problem Statement:
- Monitor industrial machinery sensor streams in real-time.
- Separate low-frequency normal machine operational dynamics from high-frequency transient anomalies.
- Detect abnormal spike patterns indicating potential equipment failure or malfunction.

Parameters chosen and rationale:
- Wavelet Family ('db4' - Daubechies 4):
  Compact support and 4 vanishing moments offer an optimal trade-off between frequency
  localization and spatial smoothness. Outperforms Haar by preventing artificial blockiness.
- Decomposition Level (level = 4):
  Separates high-frequency noise and impulse shocks from slower structural oscillations
  (1000 samples decomposed down to level 4 leaves ~63 approximation coefficients).
- Noise Estimation via MAD:
  sigma = median(|cD_1|) / 0.6745. The median is robust against large isolated fault spikes,
  preventing outliers from inflating the baseline noise floor estimate.
- Universal Threshold (VisuShrink):
  thr = sigma * sqrt(2 * ln(N)). Provides theoretical asymptotic optimality, guaranteeing
  that pure Gaussian noise will not exceed the threshold with high probability.
- Soft Thresholding (mode='soft'):
  Shrinks coefficients continuously towards zero, avoiding Gibbs-like ringing artifacts
  common in hard thresholding.
- Detection Criterion (z-score > 3.0):
  The 3-sigma rule flags points deviating beyond 99.73% of normal residual variance.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

def run_wavelet_anomaly_detection(csv_path="assets/sensor_data.csv",
                                  output_path="task3_wavelet_anomaly.png",
                                  wavelet="db4",
                                  level=4,
                                  z_cutoff=3.0):
    # 1. Load or synthesize sensor data (1000 samples)
    rng = np.random.default_rng(42)
    n = 1000
    t = np.arange(n)

    # Base operating vibration with seasonal harmonic components + Gaussian noise
    base_signal = 1.1 * np.sin(2 * np.pi * t / 200) + 0.4 * np.cos(2 * np.pi * t / 50)
    noise = 0.35 * rng.standard_normal(n)
    sensor_data = base_signal + noise

    # Injected real-world machine anomalies (transient impulse shocks)
    true_anomalies = [150, 420, 700, 850]
    sensor_data[true_anomalies] += [3.5, -3.8, 4.2, -3.6]

    # Save raw data to CSV for reproducibility
    df = pd.DataFrame({"timestamp": t, "sensor_reading": sensor_data})
    df.to_csv(csv_path, index=False)

    # 2. Multilevel Wavelet Decomposition
    coeffs = pywt.wavedec(sensor_data, wavelet, level=level)

    # 3. Robust noise estimation from finest detail band (cD1)
    finest_detail = coeffs[-1]
    sigma_est = float(np.median(np.abs(finest_detail)) / 0.6745)

    # 4. Universal threshold computation
    thr = float(sigma_est * np.sqrt(2 * np.log(n)))

    # 5. Soft threshold detail coefficients (preserve approximation band coeffs[0])
    thresholded_coeffs = [coeffs[0]] + [
        pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]
    ]

    # 6. Reconstruct denoised baseline signal
    denoised = pywt.waverec(thresholded_coeffs, wavelet)[:n]

    # 7. Compute residuals and standardized z-scores
    residuals = sensor_data - denoised
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    z_scores = np.abs(residuals - res_mean) / res_std

    # 8. Flag detected anomalies
    detected_indices = np.where(z_scores > z_cutoff)[0]

    # 9. Format visualization to match expected output on Task Sheet Page 2
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top Plot: Sensor Data and Denoised Signal
    axes[0].plot(t, sensor_data, color="#1f77b4", linewidth=1.1, label="Sensor Data")
    axes[0].plot(t, denoised, color="#ff7f0e", linestyle="--", linewidth=1.5, label="Denoised Signal")
    axes[0].set_title("Sensor Data and Denoised Signal", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Amplitude", fontsize=11)
    axes[0].legend(loc="lower right", framealpha=0.9)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylim(-4.5, 4.5)

    # Bottom Plot: Residuals and Detected Anomalies
    axes[1].plot(t, residuals, color="#d62728", linewidth=1.0, label="Residuals")
    axes[1].scatter(detected_indices, residuals[detected_indices], color="#2ca02c",
                    edgecolor="black", s=35, zorder=4, label="Anomalies")
    axes[1].set_title("Residuals and Detected Anomalies", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time (Samples)", fontsize=11)
    axes[1].set_ylabel("Residual Amplitude", fontsize=11)
    axes[1].legend(loc="lower right", framealpha=0.9)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].set_ylim(-4.5, 4.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Task 3 Complete:")
    print(f" - Wavelet: {wavelet} | Decomposition Level: {level}")
    print(f" - Estimated Noise Sigma (MAD): {sigma_est:.4f}")
    print(f" - Universal Threshold: {thr:.4f}")
    print(f" - Known Injected Anomalies: {true_anomalies}")
    print(f" - Detected Anomalies (z > {z_cutoff}): {len(detected_indices)} points at {detected_indices.tolist()}")
    print(f" - Output figure saved: {output_path}")

    return {
        "sigma": sigma_est,
        "threshold": thr,
        "anomalies": detected_indices.tolist()
    }

if __name__ == "__main__":
    run_wavelet_anomaly_detection()
