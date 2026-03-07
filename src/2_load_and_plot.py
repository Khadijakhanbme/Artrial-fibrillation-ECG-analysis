import wfdb
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# Load record
record = wfdb.rdrecord('Data/07879')

ecg_signal = record.p_signal
fs = record.fs

print("Sampling frequency:", fs)

# Plot first 5 seconds
samples = int(5 * fs)

plt.plot(ecg_signal[:samples, 0])
plt.title("ECG - First 5 Seconds")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Calculate and plot power spectral density
psd_freqs, psd_values = signal.welch(ecg_signal[:, 0], fs=fs, nperseg=1024)

plt.figure(figsize=(10, 5))
plt.semilogy(psd_freqs, psd_values)
plt.title("Power Spectral Density (Welch Method)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (V²/Hz)")
plt.grid(True, alpha=0.3)
plt.show()