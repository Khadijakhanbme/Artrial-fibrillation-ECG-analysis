import wfdb
import pandas as pd

# Load record to get fs and total length
rec = wfdb.rdrecord("Data/07879")
fs = rec.fs
N = rec.sig_len

ann = wfdb.rdann("Data/07879", "atr")

# Keep only rhythm labels like "(N", "(AFIB", etc.
rh = [(int(s), lab.strip()) for s, lab in zip(ann.sample, ann.aux_note)
      if isinstance(lab, str) and lab.startswith("(")]

# Ensure starts at 0
if len(rh) == 0 or rh[0][0] != 0:
    # assume rhythm from start is same as first known label (common practice)
    first_label = rh[0][1] if len(rh) else "(UNKNOWN"
    rh = [(0, first_label)] + rh

# Build intervals
intervals = []
for i, (start, lab) in enumerate(rh):
    end = rh[i + 1][0] if i + 1 < len(rh) else N
    typ = "AF" if lab.startswith("(AFIB") else ("Normal" if lab.startswith("(N") else "Other")
    intervals.append([typ, start, end, (end - start) / fs, start / fs, end / fs, lab])

df = pd.DataFrame(intervals, columns=["Type","Start","End","Dur_s","Start_s","End_s","RawLabel"])

af_df = df[df["Type"] == "AF"].copy()

print("Number of AF episodes:", len(af_df))
print("\nAF episodes (start_s, end_s, dur_s):")
print(af_df[["Start_s","End_s","Dur_s"]].to_string(index=False))

print("\nTotal AF duration (sec):", af_df["Dur_s"].sum())
print("Total recording duration (sec):", N / fs)

# Optional: save full timeline
df.to_csv("Data/07879_rhythm_intervals_verified.csv", index=False)
print("\nSaved timeline to Data/07879_rhythm_intervals_verified.csv")