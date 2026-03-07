import wfdb
import numpy as np

fs = 250

# Your detected peaks
rpeaks = np.load("Data/07879_rpeaks.npy")

# Reference QRS annotations (ground truth-ish)
qrs = wfdb.rdann("Data/07879", "qrs")
ref = np.array(qrs.sample)

# Match peaks within tolerance (e.g., 50 ms)
tol = int(0.05 * fs)

matched = 0
i = j = 0
while i < len(ref) and j < len(rpeaks):
    if abs(ref[i] - rpeaks[j]) <= tol:
        matched += 1
        i += 1
        j += 1
    elif rpeaks[j] < ref[i]:
        j += 1
    else:
        i += 1

recall = matched / len(ref)
precision = matched / len(rpeaks)

print("Ref QRS:", len(ref))
print("Detected:", len(rpeaks))
print("Matched:", matched)
print("Recall (miss rate):", recall, "(miss =", 1-recall, ")")
print("Precision (false+):", precision, "(false+ =", 1-precision, ")")