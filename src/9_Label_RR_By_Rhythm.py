
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 250
rpeaks = np.load("Data/07879_rpeaks.npy")

rr = np.diff(rpeaks) / fs
rr_anchor = rpeaks[:-1]  # classify RR by the first R-peak position

intervals = pd.read_csv("Data/07879_rhythm_intervals.csv")

rr_af, rr_normal = [], []
unlabeled = 0

# Efficient search: iterate intervals once (two-pointer)
i = 0
for rr_i, t in zip(rr, rr_anchor):
    while i < len(intervals) and t >= intervals.loc[i, "End"]:
        i += 1
    if i >= len(intervals):
        unlabeled += 1
        continue

    typ = intervals.loc[i, "Type"]
    if typ == "AF":
        rr_af.append(rr_i)
    elif typ == "Normal":
        rr_normal.append(rr_i)
    else:
        unlabeled += 1

rr_af = np.array(rr_af)
rr_normal = np.array(rr_normal)

print("Total RR:", len(rr))
print("AF RR:", len(rr_af))
print("Normal RR:", len(rr_normal))
print("Unlabeled/Other:", unlabeled)

np.save("Data/07879_rr_af.npy", rr_af)
np.save("Data/07879_rr_normal.npy", rr_normal)

print("Normal RR std:", np.std(rr_normal))
print("AF RR std:", np.std(rr_af))
print("Normal HR:", 60/np.mean(rr_normal))
print("AF HR:", 60/np.mean(rr_af))
