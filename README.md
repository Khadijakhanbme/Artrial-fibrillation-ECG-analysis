# Atrial Fibrillation (AF) Analysis & HRV Pipeline
### Utilizing the MIT-BIH Atrial Fibrillation Database (AFDB)

A biomedical signal processing pipeline for detecting and quantifying Atrial Fibrillation (AF) from ECG signals. Covers the full workflow from raw signal acquisition through preprocessing, R-peak detection, windowed HRV feature extraction, and rigorous statistical hypothesis testing across three patient records from the [PhysioNet AFDB](https://physionet.org/content/afdb/1.0.0/).

---

## Introduction

Atrial Fibrillation (AF) is characterized by rapid and irregular beating of the atrial chambers. This project quantifies these irregularities by analyzing ECG records **04015, 04043, and 07879** from the PhysioNet AFDB. By comparing Normal Sinus Rhythm (NSR) against AF episodes using standardized windowed HRV analysis and statistical testing, we validate the physiological impact of AF on heart rate regularity.

---

## Pipeline

An **11-step** modular Python pipeline in `/src`:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_Download_data.py` | Automated download of `.dat`, `.hea`, `.atr` files via `wfdb` |
| 2 | `2_load_and_plot.py` | Signal loading, inspection, and PSD analysis (Welch method) |
| 3 | `3_check_annotation.py` | Annotation exploration (rhythm labels, beat positions) |
| 4 | `4_Extract_Rhythm_Intervals.py` | Rhythm segmentation into Normal / AF / Other episodes |
| 5 | `5_Preprocess_ECG.py` | Baseline wander removal and noise filtering (`neurokit2`) |
| 6 | `6_R_peak_detection.py` | QRS complex detection using `neurokit2` |
| 7 | `7_Rpeak_QC.py` | R-peak validation against reference annotations (precision / recall) |
| 8 | `8_Rythm_Intervals_Verified.py` | Verified rhythm interval extraction with duration computation |
| 9 | `9_Label_RR_By_Rhythm.py` | RR interval labeling by rhythm type (two-pointer matching) |
| 10 | `10_Features.py` | Windowed HRV feature extraction across all records |
| 11 | `11_Statistical_Analysis.py` | Statistical hypothesis testing, effect size analysis, visualization |

---

## Methodology

### Signal Processing
Raw ECG signals were cleaned using `neurokit2` for baseline wander removal and noise filtering. R-peaks were detected via QRS complex identification and validated against physician-annotated reference positions using tolerance-based matching (50 ms window) to compute detection precision and recall.

### Windowed HRV Feature Extraction
HRV features are computed using **2-minute non-overlapping windows**, following short-term analysis guidelines (Task Force of ESC & NASPE, 1996). The 2-minute window was chosen to retain shorter paroxysmal AF episodes while ensuring reliable time-domain HRV estimation. Episodes shorter than 2 minutes and windows with fewer than 20 beats are excluded.

**Extracted features:**

| Feature | Description |
|---------|-------------|
| **Mean RR** | Average RR interval (s) |
| **SDNN** | Standard deviation of NN intervals - overall variability |
| **RMSSD** | Root mean square of successive differences - short-term variability |
| **pNN50** | Percentage of successive RR differences > 50 ms - beat-to-beat irregularity |
| **CV_RR** | Coefficient of variation (SDNN / Mean RR) - rate-normalized variability |
| **HR** | Heart rate (bpm) |

> **Note:** Record 04015 contributed only Normal segments, as its AF episodes were paroxysmal (<2 min), consistent with short-burst AF presentation commonly observed in clinical practice.

### Statistical Analysis

A rigorous hypothesis testing pipeline evaluates whether HRV features differ significantly between Normal and AF rhythm:

1. **Shapiro-Wilk test** - normality assessment per feature per group
2. **Levene's test** - equal variance assumption check
3. **Mann-Whitney U test** - non-parametric comparison, selected after Shapiro-Wilk rejected normality (p < 0.05) for all features due to right-skewed HRV distributions
4. **Cohen's d** - effect size quantification for practical significance

Log10 transformation was applied to SDNN, RMSSD, and CV_RR to assess whether parametric ANOVA assumptions could be met. Normality was not achieved post-transformation, confirming Mann-Whitney U as the appropriate test.

---

## Results

All five HRV features showed **statistically significant differences** (p < 0.001) between Normal and AF rhythm with large effect sizes:

| Feature | Normal (mean ± std) | AF (mean ± std) | Cohen's d | Significance |
|---------|-------------------|-----------------|-----------|-------------|
| SDNN | 0.061 ± 0.072 | 0.126 ± 0.033 | −1.01 (large) | p < 0.001 *** |
| RMSSD | 0.076 ± 0.095 | 0.175 ± 0.051 | −1.14 (large) | p < 0.001 *** |
| pNN50 | 7.07 ± 14.84 | 65.42 ± 12.87 | −3.80 (large) | p < 0.001 *** |
| CV_RR | 0.078 ± 0.087 | 0.230 ± 0.057 | −1.67 (large) | p < 0.001 *** |
| HR (bpm) | 80.81 ± 13.96 | 108.63 ± 16.53 | −1.90 (large) | p < 0.001 *** |

**Key findings:**
- AF segments exhibit significantly higher beat-to-beat variability (SDNN, RMSSD, CV_RR) and elevated heart rate compared to Normal sinus rhythm.
- **pNN50** showed the strongest discriminative power (d = −3.80), identifying beat-to-beat irregularity as the most prominent AF marker.
- The AF–Normal distinction was consistent across all patient records, supporting generalizability.

### Statistical Comparison — Normal vs AF
![Statistical Comparison](Results/Statistical%20Test%20-%20All%20records.png)

### Per-Record Breakdown
![Per-Record Comparison](Results/Per%20Subect%20HRV.png)

---

## Tech Stack

`Python` · `scipy.signal` · `neurokit2` · `wfdb` · `numpy` · `pandas` · `matplotlib` · `scipy.stats`

## Project Structure

```text
AFib_Analysis/
├── src/               # Python scripts (Steps 1–11)
├── Data/              # Records, features, statistical results
├── Results/           # Generated plots
└── requirements.txt   # Dependencies
```
