import wfdb
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load raw ECG
record = wfdb.rdrecord('Data/07879')
raw_signal = record.p_signal[:, 0]
fs = record.fs

# Clean ECG
clean_signal = nk.ecg_clean(raw_signal, sampling_rate=fs)

# -----------------------------------
# 1. Plot Raw vs Clean (10 sec)
# -----------------------------------
seconds = 10
samples = seconds * fs

plt.figure(figsize=(12,5))
plt.plot(raw_signal[:samples], label="Raw", alpha=0.6)
plt.plot(clean_signal[:samples], label="Cleaned", linewidth=2)
plt.title("Raw vs Clean ECG (First 10 sec)")
plt.legend()
plt.show()


# -----------------------------------
# 2. Power Spectrum Comparison
# -----------------------------------
f_raw, Pxx_raw = welch(raw_signal, fs, nperseg=2048)
f_clean, Pxx_clean = welch(clean_signal, fs, nperseg=2048)

plt.figure(figsize=(12,5))
plt.semilogy(f_raw, Pxx_raw, label="Raw")
plt.semilogy(f_clean, Pxx_clean, label="Cleaned")
plt.title("Power Spectral Density (Raw vs Clean)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.show()

# -----------------------------
# Save cleaned ECG
# -----------------------------
np.save("Data/07879_clean.npy", clean_signal)

print("Clean ECG saved as 07879_clean.npy")