import pandas as pd
import matplotlib.pyplot as plt

# ----- HRV results from your 3 subjects -----
data = {
    "Record": ["04015","04015","04043","04043","07879","07879"],
    "Rhythm": ["Normal","AF","Normal","AF","Normal","AF"],
    "Mean RR": [0.823,0.465,0.596,0.547,0.816,0.553],
    "SDNN": [0.132,0.116,0.077,0.134,0.094,0.151],
    "RMSSD": [0.150,0.146,0.100,0.179,0.083,0.173],
    "Heart Rate": [72.91,128.91,100.753,109.715,73.508,108.528]
}

df = pd.DataFrame(data)

# Convert to long format
df_long = df.melt(id_vars=["Record","Rhythm"],
                  var_name="Feature",
                  value_name="Value")

# Plot
plt.figure(figsize=(10,6))

for i, feature in enumerate(df_long["Feature"].unique()):
    subset = df_long[df_long["Feature"] == feature]
    
    plt.subplot(2,2,i+1)
    pivot = subset.pivot(index="Record", columns="Rhythm", values="Value")
    pivot.plot(kind="bar", ax=plt.gca())
    
    plt.title(feature)
    plt.xticks(rotation=0)

plt.suptitle("HRV Feature Comparison Across AFDB Records", fontsize=14)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

records = ["04015", "04043", "07879"]

plt.figure(figsize=(12,6))

for i, rec in enumerate(records):

    rr_normal = np.load(f"Data/{rec}_rr_normal.npy")
    rr_af = np.load(f"Data/{rec}_rr_af.npy")

    # Remove extreme pauses for cleaner visualization
    rr_normal = rr_normal[rr_normal < 2]
    rr_af = rr_af[rr_af < 2]

    x_n = rr_normal[:-1]
    y_n = rr_normal[1:]

    x_af = rr_af[:-1]
    y_af = rr_af[1:]

    # -------- Normal row --------
    plt.subplot(2, len(records), i+1)
    plt.scatter(x_n, y_n, s=4)
    plt.title(rec)
    
    if i == 0:
        plt.ylabel("Normal\nRR(n+1)")
    plt.xlabel("RR(n)")

    # -------- AF row --------
    plt.subplot(2, len(records), i+1+len(records))
    plt.scatter(x_af, y_af, s=4)

    if i == 0:
        plt.ylabel("AF\nRR(n+1)")
    plt.xlabel("RR(n)")

plt.suptitle("Poincaré Plots (AFDB Records)", fontsize=14)

plt.tight_layout()
plt.show()