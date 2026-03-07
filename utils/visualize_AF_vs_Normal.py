import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fs = 250

# Load data
ecg_clean = np.load("Data/04015_clean.npy")
rpeaks = np.load("Data/04015_rpeaks.npy")
intervals = pd.read_csv("Data/04015_rhythm_intervals_verified.csv")

window_samples = 10 * fs

# -------- NORMAL WINDOW --------
normal_row = intervals[intervals["Type"] == "Normal"].iloc[1]  # skip tiny 0-30
n_start = int(normal_row["Start"])
n_end = n_start + window_samples

rpeaks_n = rpeaks[(rpeaks >= n_start) & (rpeaks <= n_end)]

plt.figure(figsize=(12,4))
plt.plot(ecg_clean[n_start:n_end])
plt.scatter(rpeaks_n - n_start,
            ecg_clean[rpeaks_n],
            color='red')
plt.title("Normal Rhythm (10 sec)")
plt.show()


# -------- AF WINDOW --------
af_row = intervals[intervals["Type"] == "AF"].iloc[0]
a_start = int(af_row["Start"])
a_end = a_start + window_samples

rpeaks_af = rpeaks[(rpeaks >= a_start) & (rpeaks <= a_end)]

plt.figure(figsize=(12,4))
plt.plot(ecg_clean[a_start:a_end])
plt.scatter(rpeaks_af - a_start,
            ecg_clean[rpeaks_af],
            color='red')
plt.title("AF Rhythm (10 sec)")
plt.show()