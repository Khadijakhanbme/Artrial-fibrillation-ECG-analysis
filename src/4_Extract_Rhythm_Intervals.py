import wfdb
import pandas as pd

rec = wfdb.rdrecord("Data/07879")
N = rec.sig_len

ann = wfdb.rdann("Data/07879", "atr")

# Rhythm labels in aux_note usually start with "(" e.g., "(N", "(AFIB"
def is_rhythm_label(s: str) -> bool:
    return isinstance(s, str) and s.startswith("(")

rh_samples = []
rh_labels = []

for s, lab in zip(ann.sample, ann.aux_note):
    if is_rhythm_label(lab):
        rh_samples.append(int(s))
        rh_labels.append(lab.strip())

# Safety: ensure first interval starts at 0
if len(rh_samples) == 0 or rh_samples[0] > 0:
    rh_samples = [0] + rh_samples
    rh_labels = ["(UNKNOWN"] + rh_labels

# Build intervals until next rhythm label, final until end of record
intervals = []
for i in range(len(rh_samples)):
    start = rh_samples[i]
    end = rh_samples[i + 1] if i + 1 < len(rh_samples) else N
    lab = rh_labels[i]

    if lab.startswith("(AFIB"):
        typ = "AF"
    elif lab.startswith("(N"):
        typ = "Normal"
    else:
        typ = "Other"

    intervals.append([typ, start, end, lab])

df = pd.DataFrame(intervals, columns=["Type", "Start", "End", "RawLabel"])
df.to_csv("Data/07879_rhythm_intervals.csv", index=False)

print("Saved:", "Data/07879_rhythm_intervals.csv")
print(df["Type"].value_counts())