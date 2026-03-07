
# Atrial Fibrillation (AF) Analysis & HRV Pipeline
### Utilizing the MIT-BIH Atrial Fibrillation Database (AFDB)

This repository contains a robust biomedical signal processing pipeline designed to detect and analyze Atrial Fibrillation (AF) using ECG signals. The workflow covers the entire lifecycle of a medical data project, from raw data acquisition to Heart Rate Variability (HRV) feature extraction and statistical visualization.

---

## Introduction
Atrial Fibrillation (AF) is characterized by rapid and irregular beating of the atrial chambers. This project aims to quantify these irregularities by analyzing ECG records specifically **04015, 04043, and 07879** from the PhysioNet AFDB. By comparing Normal Sinus Rhythm (NSR) against AF episodes, we can visualize and measure the physiological impact on heart rate stability.

---

## The Pipeline
The project is organized into a 10-step modular Python pipeline located in the `/src` directory:

1.  **Data Acquisition**: Automated downloading of `.dat`, `.hea`, and `.atr` files via `WFDB`.
2.  **Preprocessing**: Applied baseline wander removal and noise reduction to clean the raw ECG signals.
3.  **R-Peak Detection**: Implemented QRS complex identification using the `NeuroKit2` library.
4.  **R-Peak Validation**: Cross-referenced detected peaks against reference annotations for Quality Control (QC).
5.  **Rhythm Labeling**: Segmented and labeled RR intervals as "Normal" or "AF" based on physician-verified database annotations.
6.  **Feature Extraction**: Computed HRV metrics and heart rate statistics.
7.  **Visualization**: Generated comparative plots and Poincaré maps to highlight rhythm differences.

---

## Technical Implementation
* **Signal Processing**: Used `scipy.signal` for digital filtering and signal conditioning.
* **HRV Analysis**: Calculated key time-domain metrics:
    * **Mean RR**: Average time between consecutive R-peaks.
    * **SDNN**: Standard deviation of the NN intervals (reflects overall variability).
    * **RMSSD**: Root mean square of successive differences (reflects parasympathetic activity).
* **Non-Linear Analysis**: Created **Poincaré Plots** ($RR_{n}$ vs $RR_{n+1}$) to visualize the "predictability" of the heart's rhythm.

---

## Results & Observations
The analysis revealed clear physiological markers for Atrial Fibrillation:
* **Variability**: AF segments showed significantly higher **SDNN** and **RMSSD** values compared to normal rhythm, indicating high irregularity.
* **Heart Rate**: AF episodes were characterized by shorter RR intervals and a noticeably higher average Heart Rate (HR).
* **Poincaré Patterns**: 
    * **Normal Rhythm**: Displays a tight, cigar-shaped cluster along the identity line.
    * **AF Rhythm**: Displays a scattered "cloud" or "fan" distribution, representing the stochastic nature of the irregular intervals.

---

## Project Structure
```text
AFib Analysis/
├── src/            # 1-10 Step-by-step Python scripts
├── utils/          # Visualization scripts
├── Results/        # Generated plots
└── requirements.txt # Project dependencies
