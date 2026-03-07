import numpy as np
import neurokit2 as nk

fs = 250

# Load cleaned ECG
ecg_clean = np.load("Data/07879_clean.npy")

print("Detecting R-peaks...")

# Detect peaks ONLY (no re-cleaning)
_, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

rpeaks = info["ECG_R_Peaks"]

# Save R-peaks
np.save("Data/07879_rpeaks.npy", rpeaks)

print("Total R-peaks detected:", len(rpeaks))
print("Saved R-peaks.")

#Quick Validation Check
avg_hr = (len(rpeaks) / (36823)) * 60
print("Average HR (bpm):", avg_hr)