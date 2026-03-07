import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load RR intervals
rr_af = np.load("Data/07879_rr_af.npy")
rr_normal = np.load("Data/07879_rr_normal.npy")

# ---------- Feature function ----------
def compute_features(rr):
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)                 # SDNN
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    hr = 60 / mean_rr

    return {
        "Mean RR (s)": mean_rr,
        "SDNN": std_rr,
        "RMSSD": rmssd,
        "Heart Rate (bpm)": hr,
        "Num Beats": len(rr)
    }

# Compute features
af_features = compute_features(rr_af)
normal_features = compute_features(rr_normal)

# Create comparison table
df = pd.DataFrame([normal_features, af_features],
                  index=["Normal", "AF"])

print("\nHRV Feature Comparison\n")
print(df.round(3))

# ---------- Plot ----------
df[["Mean RR (s)", "SDNN", "RMSSD"]].plot(kind="bar", figsize=(8,5))
plt.title("HRV Feature Comparison: Normal vs AF")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()