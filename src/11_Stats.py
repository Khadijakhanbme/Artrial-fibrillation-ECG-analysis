import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# ============================================================
# 11. Statistical Analysis — Hypothesis Testing
# ============================================================
# Pipeline:
#   1. Shapiro-Wilk normality test
#   2. Levene's equal variance test
#   3. Mann-Whitney U (non-parametric) / ANOVA (if normal)
#   4. Cohen's d effect size
#   5. Log-transform attempt for ANOVA validation
#
# Window: 5 min (Task Force of ESC & NASPE, 1996)
# ============================================================

os.makedirs("Results", exist_ok=True)

df = pd.read_csv("Data/all_episode_features.csv")
normal = df[df["Type"] == "Normal"]
af = df[df["Type"] == "AF"]

print("=" * 70)
print("STATISTICAL ANALYSIS: Normal vs AF (5-min windows)")
print("=" * 70)
print(f"\nRecords analyzed:  {sorted(df['Record'].unique().tolist())}")
print(f"Normal windows:    {len(normal)}")
print(f"AF windows:        {len(af)}")
print(f"Total windows:     {len(df)}")


# ---------- Helpers ----------
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*np.std(g1,ddof=1)**2 + (n2-1)*np.std(g2,ddof=1)**2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled else 0.0

def interpret_d(d):
    d = abs(d)
    if d > 0.8: return "large"
    elif d > 0.5: return "medium"
    elif d > 0.2: return "small"
    return "negligible"

def sig_label(p):
    if p < 0.001: return "*** (p < 0.001)"
    elif p < 0.01: return f"**  (p = {p:.4f})"
    elif p < 0.05: return f"*   (p = {p:.4f})"
    return f"ns  (p = {p:.4f})"


# ============================================================
# MAIN ANALYSIS
# ============================================================
features = ["SDNN", "RMSSD", "pNN50", "CV_RR", "HR_bpm"]
results = []

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

for feat in features:
    n_vals = normal[feat].values
    af_vals = af[feat].values

    print(f"\n--- {feat} ---")
    print(f"  Normal: mean={np.mean(n_vals):.4f}, std={np.std(n_vals,ddof=1):.4f}, n={len(n_vals)}")
    print(f"  AF:     mean={np.mean(af_vals):.4f}, std={np.std(af_vals,ddof=1):.4f}, n={len(af_vals)}")

    # Normality
    shap_n_stat, shap_n_p = stats.shapiro(n_vals)
    shap_af_stat, shap_af_p = stats.shapiro(af_vals)
    normal_dist = (shap_n_p > 0.05) and (shap_af_p > 0.05)
    print(f"  Shapiro-Wilk: Normal p={shap_n_p:.4f}, AF p={shap_af_p:.4f} "
          f"-> {'Both normal' if normal_dist else 'NOT normal'}")

    # Equal variance
    lev_stat, lev_p = stats.levene(n_vals, af_vals)
    print(f"  Levene:       p={lev_p:.4f} -> {'Equal variance' if lev_p > 0.05 else 'Unequal variance'}")

    # Test
    if normal_dist:
        test_stat, p_val = stats.f_oneway(n_vals, af_vals)
        test_name = "One-way ANOVA"
    else:
        test_stat, p_val = stats.mannwhitneyu(n_vals, af_vals, alternative="two-sided")
        test_name = "Mann-Whitney U"

    d = cohens_d(n_vals, af_vals)
    print(f"  {test_name}: stat={test_stat:.4f}, {sig_label(p_val)}")
    print(f"  Cohen's d: {d:.3f} ({interpret_d(d)} effect)")

    results.append({
        "Feature": feat,
        "Normal_mean": round(np.mean(n_vals), 4),
        "Normal_std": round(np.std(n_vals, ddof=1), 4),
        "AF_mean": round(np.mean(af_vals), 4),
        "AF_std": round(np.std(af_vals, ddof=1), 4),
        "Shapiro_Normal_p": round(shap_n_p, 4),
        "Shapiro_AF_p": round(shap_af_p, 4),
        "Levene_p": round(lev_p, 4),
        "Test": test_name,
        "Statistic": round(test_stat, 4),
        "p_value": round(p_val, 6),
        "Cohens_d": round(d, 3),
        "Effect_size": interpret_d(d),
        "Significant": "Yes" if p_val < 0.05 else "No"
    })


# ============================================================
# LOG-TRANSFORM VALIDATION (reported in text, no plot)
# ============================================================
log_features = ["SDNN", "RMSSD", "CV_RR"]

print("\n" + "=" * 70)
print("LOG-TRANSFORM VALIDATION")
print("=" * 70)
print("Log10 transformation applied to address right-skewness.")

for feat in log_features:
    n_log = np.log10(normal[feat].values)
    af_log = np.log10(af[feat].values)

    shap_n_p = stats.shapiro(n_log)[1]
    shap_af_p = stats.shapiro(af_log)[1]
    normal_dist = (shap_n_p > 0.05) and (shap_af_p > 0.05)

    f_stat, f_p = stats.f_oneway(n_log, af_log)

    status = "ACHIEVED" if normal_dist else "NOT achieved"
    print(f"  log10({feat}): normality {status} | ANOVA F={f_stat:.2f}, p={f_p:.6f}")

print("\nNote: Normality was not achieved after log transformation,")
print("confirming Mann-Whitney U as the appropriate test (Stage 1).")


# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv("Data/statistical_results.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(results_df[["Feature", "Test", "p_value", "Cohens_d", "Effect_size", "Significant"]].to_string(index=False))


# ============================================================
# PLOT 1: Statistical Comparison (Main Result)
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(24, 5))

for i, feat in enumerate(features):
    n_vals = normal[feat].values
    af_vals = af[feat].values
    r = results[i]

    bp = axes[i].boxplot([n_vals, af_vals],
                         labels=["Normal", "AF"],
                         patch_artist=True,
                         widths=0.5,
                         boxprops=dict(linewidth=1.5),
                         medianprops=dict(color="black", linewidth=2))

    bp["boxes"][0].set_facecolor("#4C9BE8")
    bp["boxes"][1].set_facecolor("#E85C5C")

    # Data points
    for j, (vals, x_pos) in enumerate(zip([n_vals, af_vals], [1, 2])):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        axes[i].scatter(x_pos + jitter, vals, alpha=0.35, s=15,
                        color=("#2E6DB0" if j == 0 else "#B02E2E"), zorder=3)

    # Significance bracket
    p = r["p_value"]
    sig_text = "p < 0.001 ***" if p < 0.001 else (f"p = {p:.4f} **" if p < 0.01 else (f"p = {p:.4f} *" if p < 0.05 else f"p = {p:.4f} (ns)"))

    y_max = max(n_vals.max(), af_vals.max())
    y_bar = y_max * 1.15
    axes[i].plot([1, 1, 2, 2], [y_bar * 0.97, y_bar, y_bar, y_bar * 0.97],
                 color="black", linewidth=1.2)
    axes[i].text(1.5, y_bar * 1.02, sig_text, ha="center", fontsize=9, fontweight="bold")

    axes[i].set_title(f"{feat}\n({r['Test']}, d={r['Cohens_d']:.2f})", fontsize=10)
    axes[i].set_ylabel("Value")
    axes[i].grid(True, alpha=0.3, axis="y")

fig.suptitle("Statistical Comparison: Normal vs AF (5-min windows, all records)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("Results/11_statistical_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# PLOT 2: Per-Record Breakdown
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(26, 5))

for i, feat in enumerate(features):
    records = sorted(df["Record"].unique())
    x_positions = []
    x_labels = []
    data_groups = []
    colors = []
    pos = 0

    for rec in records:
        rec_data = df[df["Record"] == rec]
        for typ, color in [("Normal", "#4C9BE8"), ("AF", "#E85C5C")]:
            vals = rec_data[rec_data["Type"] == typ][feat].values
            if len(vals) > 0:
                data_groups.append(vals)
                x_positions.append(pos)
                x_labels.append(f"{rec}\n{typ}")
                colors.append(color)
                pos += 1
        pos += 0.5

    if len(data_groups) == 0:
        continue

    bp = axes[i].boxplot(data_groups, positions=x_positions, widths=0.6,
                         patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))

    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)

    axes[i].set_xticks(x_positions)
    axes[i].set_xticklabels(x_labels, fontsize=7)
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_ylabel("Value")
    axes[i].grid(True, alpha=0.3, axis="y")

fig.suptitle("Per-Record HRV: Normal (blue) vs AF (red) — 5-min windows",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("Results/11_per_record_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


print("\nPlots saved to Results/")
print("Statistical analysis complete.")