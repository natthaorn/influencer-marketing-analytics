# =============================================================================
# INFLUENCER MARKETING ANALYTICS — Phase 3: Clustering & ROI Scoring
# Senior Data Analyst Portfolio Project
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # no popups — save all figures to disk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import zscore

# ── Global style (matches Phase 2) ───────────────────────────────────────────
PALETTE    = ["#E1306C", "#833AB4", "#405DE6", "#F77737", "#FCAF45"]
BG_COLOR   = "#0F0F0F"
TEXT_COLOR = "#F5F5F5"
sns.set_theme(style="darkgrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   "#1A1A1A",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       "#2A2A2A",
    "font.family":      "DejaVu Sans",
})

import os
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/data",  exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{name}.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  ✅ saved → outputs/plots/{name}.png")


# =============================================================================
# 1. LOAD & PREPARE DATA  (reuse cleaning from Phase 2)
# =============================================================================
PATH_INFLUENCERS = "data/raw/top_insta_influencers_data.csv"

df_raw = pd.read_csv(PATH_INFLUENCERS)

# ── Rename ────────────────────────────────────────────────────────────────────
rename_map = {
    "channel_info":      "username",
    "influence_score":   "influence_score",
    "posts":             "posts",
    "followers":         "followers",
    "avg_likes":         "avg_likes",
    "60_day_eng_rate":   "eng_rate_60d",
    "new_post_avg_like": "new_post_avg_likes",
    "total_likes":       "total_likes",
    "country":           "country",
    "topic":             "topic",
}
df = df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns})

# ── Parse suffix numbers (1.2M → 1_200_000) ──────────────────────────────────
def parse_num(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", "")
    for suffix, mult in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if s.upper().endswith(suffix):
            return float(s[:-1]) * mult
    try: return float(s)
    except: return np.nan

for col in ["followers", "avg_likes", "new_post_avg_likes", "total_likes", "posts"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_num)

if "eng_rate_60d" in df.columns:
    df["eng_rate_60d"] = (
        df["eng_rate_60d"].astype(str)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

df.drop_duplicates(inplace=True)
df.dropna(subset=["followers", "avg_likes"], inplace=True)

# ── Feature engineering ───────────────────────────────────────────────────────
df["eng_rate_computed"] = (df["avg_likes"] / df["followers"]) * 100
df["log_followers"]     = np.log10(df["followers"].replace(0, np.nan))
df["log_avg_likes"]     = np.log10(df["avg_likes"].replace(0, np.nan))
df["log_total_likes"]   = np.log10(df["total_likes"].replace(0, np.nan)) \
                          if "total_likes" in df.columns else np.nan

# ── Updated tier segmentation ─────────────────────────────────────────────────
def assign_tier(f):
    if f < 10_000_000:   return "Rising Mega (1M–10M)"
    elif f < 50_000_000: return "Established Mega (10M–50M)"
    elif f < 100_000_000:return "Super Mega (50M–100M)"
    else:                return "Icon (100M+)"

TIER_ORDER = ["Rising Mega (1M–10M)", "Established Mega (10M–50M)",
              "Super Mega (50M–100M)", "Icon (100M+)"]
df["tier"] = df["followers"].apply(assign_tier)

print(f"✅ Data loaded: {len(df):,} influencers")
print(df[["username","followers","avg_likes","eng_rate_computed","tier"]].head())


# =============================================================================
# 2. SELECT CLUSTERING FEATURES
# =============================================================================
CLUSTER_FEATURES = [c for c in [
    "log_followers",
    "eng_rate_computed",
    "log_avg_likes",
    "influence_score",
    "posts",
] if c in df.columns]

df_cluster = df[CLUSTER_FEATURES].dropna()
df_model   = df.loc[df_cluster.index].copy()   # aligned full df

print(f"\n🔧 Clustering features : {CLUSTER_FEATURES}")
print(f"   Rows used           : {len(df_cluster):,}")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)


# =============================================================================
# 3. FIND OPTIMAL K  (Elbow + Silhouette)
# =============================================================================
K_RANGE   = range(2, 9)
inertias  = []
sil_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal K Selection", fontsize=16, fontweight="bold", color=TEXT_COLOR)

# Elbow
axes[0].plot(list(K_RANGE), inertias, "o-", color=PALETTE[0], linewidth=2, markersize=8)
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")

# Silhouette
axes[1].plot(list(K_RANGE), sil_scores, "s-", color=PALETTE[1], linewidth=2, markersize=8)
best_k = list(K_RANGE)[np.argmax(sil_scores)]
axes[1].axvline(best_k, color=PALETTE[3], linestyle="--", linewidth=1.5,
                label=f"Best k = {best_k}")
axes[1].set_title("Silhouette Score (higher = better)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend(labelcolor=TEXT_COLOR, framealpha=0.3)

savefig("09_optimal_k")
print(f"\n📌 Best k by silhouette = {best_k}")


# =============================================================================
# 4. FIT FINAL K-MEANS MODEL
# =============================================================================
K_FINAL = best_k
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df_model["cluster"] = km_final.fit_predict(X_scaled)

# ── Cluster summary ───────────────────────────────────────────────────────────
cluster_summary = (
    df_model.groupby("cluster")[CLUSTER_FEATURES + ["eng_rate_computed","followers"]]
    .mean()
    .round(3)
)
cluster_summary["count"] = df_model.groupby("cluster").size()
print("\n📊 Cluster Summary (means):")
print(cluster_summary.to_string())


# =============================================================================
# 5. LABEL CLUSTERS (auto-label by engagement & followers)
# =============================================================================
# Sort clusters: high engagement + lower followers = "Engagement King"
#                high followers + lower engagement = "Reach Giant"
#                high both                         = "Power Influencer"
#                lower both                        = "Niche Player"

def auto_label(row):
    eng  = row["eng_rate_computed"]
    flw  = row["followers"]
    eng_med = df_model["eng_rate_computed"].median()
    flw_med = df_model["followers"].median()
    if eng >= eng_med and flw >= flw_med:
        return "⚡ Power Influencer"
    elif eng >= eng_med and flw < flw_med:
        return "💬 Engagement King"
    elif eng < eng_med and flw >= flw_med:
        return "📡 Reach Giant"
    else:
        return "🎯 Niche Player"

cluster_means = df_model.groupby("cluster")[["eng_rate_computed","followers"]].mean()
cluster_label_map = {idx: auto_label(row) for idx, row in cluster_means.iterrows()}
df_model["cluster_label"] = df_model["cluster"].map(cluster_label_map)

print("\n🏷️  Cluster Labels:")
for k, v in cluster_label_map.items():
    n = (df_model["cluster"] == k).sum()
    print(f"   Cluster {k} → {v}  (n={n})")


# =============================================================================
# 6. VISUALISE CLUSTERS
# =============================================================================

# ── 6a. PCA 2D Scatter ────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_model["pca1"] = X_pca[:, 0]
df_model["pca2"] = X_pca[:, 1]

unique_labels = df_model["cluster_label"].unique()
color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(unique_labels)}

fig, ax = plt.subplots(figsize=(12, 7))
for label, grp in df_model.groupby("cluster_label"):
    ax.scatter(grp["pca1"], grp["pca2"],
               c=color_map[label], label=label,
               alpha=0.75, s=60, edgecolors="none")

ax.set_title("K-Means Clusters — PCA 2D View\n"
             f"(explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance)",
             fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(title="Cluster", framealpha=0.3, labelcolor=TEXT_COLOR)

savefig("10_cluster_pca_scatter")


# ── 6b. Cluster Profile — Radar / Bar ────────────────────────────────────────
profile_cols = [c for c in ["log_followers","eng_rate_computed","log_avg_likes",
                             "influence_score"] if c in df_model.columns]

cluster_profile = df_model.groupby("cluster_label")[profile_cols].mean()
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / \
                       (cluster_profile.max() - cluster_profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(profile_cols))
width = 0.8 / len(cluster_profile_norm)

for i, (label, row) in enumerate(cluster_profile_norm.iterrows()):
    bars = ax.bar(x + i * width, row.values, width,
                  label=label, color=PALETTE[i % len(PALETTE)], alpha=0.85)

ax.set_xticks(x + width * (len(cluster_profile_norm) - 1) / 2)
ax.set_xticklabels([c.replace("_", " ").replace("log ", "log\n")
                    for c in profile_cols], fontsize=10)
ax.set_ylabel("Normalised Score (0–1)")
ax.set_title("Cluster Performance Profiles\n(normalised for comparison)",
             fontsize=14, fontweight="bold")
ax.legend(title="Cluster", framealpha=0.3, labelcolor=TEXT_COLOR, fontsize=9)
ax.set_ylim(0, 1.15)

savefig("11_cluster_profiles")


# ── 6c. Engagement vs Followers coloured by cluster ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for label, grp in df_model.groupby("cluster_label"):
    ax.scatter(grp["log_followers"],
               grp["eng_rate_computed"].clip(
                   upper=grp["eng_rate_computed"].quantile(0.97)),
               c=color_map[label], label=label,
               alpha=0.7, s=55, edgecolors="none")

ax.set_title("Followers vs Engagement Rate — by Cluster",
             fontsize=14, fontweight="bold")
ax.set_xlabel("log₁₀(Followers)")
ax.set_ylabel("Engagement Rate (%)")
ax.legend(title="Cluster", framealpha=0.3, labelcolor=TEXT_COLOR)

savefig("12_cluster_eng_vs_followers")


# ── 6d. Cluster Size Bar ──────────────────────────────────────────────────────
size_series = df_model["cluster_label"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(size_series.index, size_series.values,
              color=[color_map[l] for l in size_series.index], alpha=0.85)
ax.set_title("Influencer Count per Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Count")
for bar, val in zip(bars, size_series.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", fontsize=11, color=TEXT_COLOR, fontweight="bold")

savefig("13_cluster_sizes")


# =============================================================================
# 7. ROI SCORING MODEL
# =============================================================================
# Composite score = weighted sum of normalised metrics
# Weights reflect what brands care about most

WEIGHTS = {
    "eng_rate_computed": 0.40,   # engagement = most important
    "influence_score":   0.25,   # overall influence
    "log_avg_likes":     0.20,   # content performance
    "log_followers":     0.15,   # reach
}
WEIGHTS = {k: v for k, v in WEIGHTS.items() if k in df_model.columns}

# Normalise each feature 0–1
df_score = df_model.copy()
for feat in WEIGHTS:
    col_min = df_score[feat].min()
    col_max = df_score[feat].max()
    df_score[f"{feat}_norm"] = (df_score[feat] - col_min) / (col_max - col_min + 1e-9)

# Weighted sum
df_score["roi_score"] = sum(
    df_score[f"{feat}_norm"] * w for feat, w in WEIGHTS.items()
)
df_score["roi_score"] = (df_score["roi_score"] * 100).round(2)  # scale 0–100

# ── Score tiers ───────────────────────────────────────────────────────────────
def score_grade(s):
    if s >= 75: return "🥇 Tier A — Premium"
    elif s >= 50: return "🥈 Tier B — Strong"
    elif s >= 25: return "🥉 Tier C — Moderate"
    else:         return "⚪ Tier D — Low"

df_score["score_grade"] = df_score["roi_score"].apply(score_grade)

print("\n📊 ROI Score Distribution:")
print(df_score["score_grade"].value_counts().to_string())
print(f"\n   Top scorer  : {df_score.loc[df_score['roi_score'].idxmax(), 'username']} "
      f"— {df_score['roi_score'].max():.1f}/100")
print(f"   Mean score  : {df_score['roi_score'].mean():.1f}/100")


# ── 7a. ROI Score Distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df_score["roi_score"], bins=30, color=PALETTE[0], edgecolor="none", alpha=0.85)
for thresh, label, color in [(75, "Tier A", PALETTE[3]),
                              (50, "Tier B", PALETTE[1]),
                              (25, "Tier C", PALETTE[2])]:
    ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5, label=label)
ax.set_title("ROI Score Distribution (0–100)", fontsize=14, fontweight="bold")
ax.set_xlabel("ROI Score")
ax.set_ylabel("Count")
ax.legend(title="Grade Threshold", framealpha=0.3, labelcolor=TEXT_COLOR)

savefig("14_roi_score_distribution")


# ── 7b. Top 20 Influencers by ROI Score ──────────────────────────────────────
top20 = df_score.nlargest(20, "roi_score")[["username","roi_score","score_grade",
                                             "followers","eng_rate_computed",
                                             "cluster_label"]]

fig, ax = plt.subplots(figsize=(13, 8))
colors = [PALETTE[0] if g.startswith("🥇") else
          PALETTE[1] if g.startswith("🥈") else PALETTE[2]
          for g in top20["score_grade"]]

bars = ax.barh(top20["username"], top20["roi_score"], color=colors, alpha=0.85)
ax.set_title("Top 20 Influencers by ROI Score", fontsize=14, fontweight="bold")
ax.set_xlabel("ROI Score (0–100)")
ax.invert_yaxis()
for bar, (_, row) in zip(bars, top20.iterrows()):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{row['roi_score']:.1f}  |  {row['eng_rate_computed']:.2f}% eng  "
            f"|  {row['cluster_label']}",
            va="center", fontsize=8, color=TEXT_COLOR)

legend_patches = [
    mpatches.Patch(color=PALETTE[0], label="🥇 Tier A — Premium"),
    mpatches.Patch(color=PALETTE[1], label="🥈 Tier B — Strong"),
    mpatches.Patch(color=PALETTE[2], label="🥉 Tier C — Moderate"),
]
ax.legend(handles=legend_patches, framealpha=0.3, labelcolor=TEXT_COLOR, fontsize=9)

savefig("15_top20_roi_leaderboard")


# ── 7c. ROI Score by Cluster ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=df_score, x="cluster_label", y="roi_score",
            palette=PALETTE[:K_FINAL], ax=ax,
            flierprops=dict(marker="o", markersize=3, alpha=0.4))
ax.set_title("ROI Score Distribution by Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("ROI Score (0–100)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

for i, label in enumerate(df_score["cluster_label"].unique()):
    med = df_score[df_score["cluster_label"] == label]["roi_score"].median()
    ax.text(i, med + 0.5, f"{med:.1f}", ha="center",
            fontsize=9, color="white", fontweight="bold")

savefig("16_roi_by_cluster")


# =============================================================================
# 8. EXPORT RESULTS
# =============================================================================
output_cols = ["username", "followers", "avg_likes", "eng_rate_computed",
               "tier", "cluster", "cluster_label", "roi_score", "score_grade"]
if "country" in df_score.columns:  output_cols.append("country")
if "topic"   in df_score.columns:  output_cols.append("topic")

output_cols = [c for c in output_cols if c in df_score.columns]
df_export = df_score[output_cols].sort_values("roi_score", ascending=False)
df_export.to_csv("outputs/data/influencer_scored.csv", index=False)

print("\n" + "="*60)
print("  📦 PHASE 3 COMPLETE")
print("="*60)
print(f"\n  Plots saved  → outputs/plots/  (09 to 16)")
print(f"  Data saved   → outputs/data/influencer_scored.csv")
print(f"\n  Clusters found    : {K_FINAL}")
print(f"  Influencers scored: {len(df_export):,}")
print(f"\n  Cluster breakdown:")
print(df_score["cluster_label"].value_counts().to_string())
print(f"\n  Score grade breakdown:")
print(df_score["score_grade"].value_counts().to_string())
print("\n  ✅ Next → Phase 4: Dashboard (Tableau / Power BI)")
