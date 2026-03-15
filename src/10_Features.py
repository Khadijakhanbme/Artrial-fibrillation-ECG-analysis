import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("Results", exist_ok=True)

fs = 250
WINDOW_SEC = 300        # 5 minutes
WINDOW_SAMPLES = WINDOW_SEC * fs
MIN_BEATS = 20          # minimum beats within a 5-min window

RECORDS = ["04015", "04043", "07879"]


# ---------- Feature function ----------
def compute_features(rr):
    """Compute HRV time-domain features from RR interval array."""
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    hr = 60.0 / mean_rr

    # pNN50: percentage of successive RR differences > 50ms
    nn_diffs = np.abs(np.diff(rr))
    pnn50 = (np.sum(nn_diffs > 0.05) / len(nn_diffs)) * 100

    # CV_RR: Coefficient of Variation (normalizes variability by heart rate)
    cv_rr = sdnn / mean_rr

    return {
        "Mean_RR": round(mean_rr, 4),
        "SDNN": round(sdnn, 4),
        "RMSSD": round(rmssd, 4),
        "pNN50": round(pnn50, 2),
        "CV_RR": round(cv_rr, 4),
        "HR_bpm": round(hr, 2),
        "Num_Beats": len(rr),
    }


# ---------- Loop over all records ----------
all_rows = []
skipped_short_episodes = 0
skipped_short_windows = 0

for rec_id in RECORDS:
    rpeaks_path = f"Data/{rec_id}_rpeaks.npy"
    intervals_path = f"Data/{rec_id}_rhythm_intervals.csv"

    if not os.path.exists(rpeaks_path) or not os.path.exists(intervals_path):
        print(f"[SKIP] Missing files for record {rec_id}")
        continue

    rpeaks = np.load(rpeaks_path)
    intervals = pd.read_csv(intervals_path)

    print(f"\n--- Record {rec_id} ---")
    print(f"  Episodes in CSV: AF={len(intervals[intervals['Type']=='AF'])}  "
          f"Normal={len(intervals[intervals['Type']=='Normal'])}  "
          f"Other={len(intervals[intervals['Type']=='Other'])}")

    for idx, row in intervals.iterrows():
        typ = row["Type"]
        if typ not in ("AF", "Normal"):
            continue

        start = int(row["Start"])
        end = int(row["End"])
        episode_duration_s = (end - start) / fs

        # Skip episodes shorter than 5 minutes
        if episode_duration_s < WINDOW_SEC:
            skipped_short_episodes += 1
            continue

        # Slide 5-minute windows through this episode
        win_start = start
        win_num = 0

        while (win_start + WINDOW_SAMPLES) <= end:
            win_end = win_start + WINDOW_SAMPLES

            # Get R-peaks within this 5-min window
            mask = (rpeaks >= win_start) & (rpeaks < win_end)
            win_peaks = rpeaks[mask]

            if len(win_peaks) < MIN_BEATS:
                skipped_short_windows += 1
                win_start = win_end
                continue

            # Compute RR intervals and features
            rr = np.diff(win_peaks) / fs
            feats = compute_features(rr)

            feats["Record"] = rec_id
            feats["Type"] = typ
            feats["Episode"] = idx
            feats["Window"] = win_num
            feats["Win_Start_s"] = round(win_start / fs, 2)
            feats["Win_End_s"] = round(win_end / fs, 2)
            all_rows.append(feats)

            win_num += 1
            win_start = win_end

    print(f"  Windows extracted so far: {len(all_rows)}")

# ---------- Build DataFrame ----------
df = pd.DataFrame(all_rows)

col_order = ["Record", "Episode", "Window", "Type", "Win_Start_s", "Win_End_s",
             "Num_Beats", "Mean_RR", "SDNN", "RMSSD", "pNN50", "CV_RR", "HR_bpm"]
df = df[col_order]

n_normal = len(df[df["Type"] == "Normal"])
n_af = len(df[df["Type"] == "AF"])

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total 5-min windows:      {len(df)}")
print(f"  Normal windows:         {n_normal}")
print(f"  AF windows:             {n_af}")
print(f"Skipped episodes (<5min): {skipped_short_episodes}")
print(f"Skipped windows (<{MIN_BEATS} beats): {skipped_short_windows}")

print("\n===== Sample of extracted features =====\n")
print(df.head(15).to_string(index=False))

# ---------- Descriptive Statistics ----------
print("\n===== Descriptive Statistics by Group =====\n")
desc = df.groupby("Type")[["SDNN", "RMSSD", "pNN50", "CV_RR", "HR_bpm"]].agg(
    ["count", "mean", "std", "min", "max"])
print(desc.round(4).to_string())

# ---------- Save ----------
df.to_csv("Data/all_episode_features.csv", index=False)
print("\nSaved: Data/all_episode_features.csv")


# ============================================================
# VISUALIZATIONS
# ============================================================

# 1. Bar chart: Mean feature values per group
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: time-domain features (seconds scale)
summary1 = df.groupby("Type")[["Mean_RR", "SDNN", "RMSSD"]].mean()
summary1.plot(kind="bar", ax=axes[0])
axes[0].set_title("Mean HRV Features (seconds)")
axes[0].set_ylabel("Value (seconds)")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Right: pNN50 (%) and CV_RR (ratio) and HR (bpm) — different scales
summary2 = df.groupby("Type")[["pNN50", "CV_RR"]].mean()
summary2.plot(kind="bar", ax=axes[1])
axes[1].set_title("pNN50 (%) and CV_RR (ratio)")
axes[1].set_ylabel("Value")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

fig.suptitle("Mean HRV Features: Normal vs AF (5-min windows, all records)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("Results/10_mean_features_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. Boxplots for all features
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for i, feat in enumerate(["SDNN", "RMSSD", "pNN50", "CV_RR", "HR_bpm"]):
    normal_vals = df[df["Type"] == "Normal"][feat]
    af_vals = df[df["Type"] == "AF"][feat]

    bp = axes[i].boxplot([normal_vals, af_vals],
                         labels=["Normal", "AF"],
                         patch_artist=True,
                         boxprops=dict(facecolor="lightblue"),
                         medianprops=dict(color="red", linewidth=2))
    bp["boxes"][1].set_facecolor("salmon")
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_ylabel("Value")
    axes[i].grid(True, alpha=0.3)

fig.suptitle("HRV Feature Distributions: Normal vs AF (5-min windows, all records)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Results/10_feature_boxplots.png", dpi=150)
plt.show()

# 3. Feature histograms
fig, axes = plt.subplots(1, 5, figsize=(24, 5))

for i, feat in enumerate(["SDNN", "RMSSD", "pNN50", "CV_RR", "HR_bpm"]):
    normal_vals = df[df["Type"] == "Normal"][feat]
    af_vals = df[df["Type"] == "AF"][feat]

    axes[i].hist(normal_vals, bins=15, alpha=0.6, color="#4C9BE8", label="Normal", edgecolor="white")
    axes[i].hist(af_vals, bins=15, alpha=0.6, color="#E85C5C", label="AF", edgecolor="white")
    axes[i].set_title(feat)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count (windows)")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.suptitle("Feature Distributions: Normal vs AF (5-min windows, all records)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Results/10_feature_histograms.png", dpi=150)
plt.show()

print("\nPlots saved to Results/")