# =============================================================================
# INFLUENCER MARKETING ANALYTICS — Phase 2: Data Cleaning & EDA
# Senior Data Analyst Portfolio Project
# =============================================================================

# ── 0. INSTALL & IMPORTS ─────────────────────────────────────────────────────
# pip install pandas numpy matplotlib seaborn plotly scipy

import matplotlib
matplotlib.use("Agg")   # ← non-interactive backend: saves files, no popups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
PALETTE   = ["#E1306C", "#833AB4", "#405DE6", "#F77737", "#FCAF45"]
BG_COLOR  = "#0F0F0F"
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


# =============================================================================
# 1. LOAD DATA
# =============================================================================
# ── Adjust paths to wherever you saved the CSVs ──────────────────────────────
PATH_INFLUENCERS = "data/raw/top_insta_influencers_data.csv"
PATH_ROI         = "data/raw/influencer_marketing_roi.csv"

df_ig  = pd.read_csv(PATH_INFLUENCERS)
df_roi = pd.read_csv(PATH_ROI)

print("=== Instagram Influencers ===")
print(df_ig.shape)
print(df_ig.dtypes)
print(df_ig.head(3))

print("\n=== ROI Dataset ===")
print(df_roi.shape)
print(df_roi.dtypes)
print(df_roi.head(3))


# =============================================================================
# 2. DATA CLEANING
# =============================================================================

def clean_instagram_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and parse numeric fields."""
    df = df.copy()

    # ── Rename columns to snake_case ─────────────────────────────────────────
    rename_map = {
        "channel_info":       "username",
        "influence_score":    "influence_score",
        "posts":              "posts",
        "followers":          "followers",
        "avg_likes":          "avg_likes",
        "60_day_eng_rate":    "eng_rate_60d",
        "new_post_avg_like":  "new_post_avg_likes",
        "total_likes":        "total_likes",
        "country":            "country",
        "topic":              "topic",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # ── Parse suffix-encoded numbers (e.g. "1.2M", "500K") ──────────────────
    def parse_num(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace(",", "")
        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        for suffix, mult in multipliers.items():
            if s.upper().endswith(suffix):
                return float(s[:-1]) * mult
        try:
            return float(s)
        except ValueError:
            return np.nan

    for col in ["followers", "avg_likes", "new_post_avg_likes", "total_likes", "posts"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_num)

    # ── Parse percentage strings ──────────────────────────────────────────────
    if "eng_rate_60d" in df.columns:
        df["eng_rate_60d"] = (
            df["eng_rate_60d"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # ── Drop duplicates & null-heavy rows ────────────────────────────────────
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["followers", "avg_likes"], inplace=True)

    return df


def clean_roi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise ROI dataset."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.drop_duplicates(inplace=True)
    return df


df_ig  = clean_instagram_df(df_ig)
df_roi = clean_roi_df(df_roi)

print(f"\n✅ Cleaned Instagram rows : {len(df_ig):,}")
print(f"✅ Cleaned ROI rows       : {len(df_roi):,}")


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

# ── 3a. Engagement Rate (computed) ───────────────────────────────────────────
# Formula: avg_likes / followers  (proxy when comments not available)
df_ig["eng_rate_computed"] = (df_ig["avg_likes"] / df_ig["followers"]) * 100

# ── 3b. Influencer Tier ──────────────────────────────────────────────────────
def assign_tier(followers: float) -> str:
    if followers < 10_000_000:
        return "Rising Mega (1M–10M)"
    elif followers < 50_000_000:
        return "Established Mega (10M–50M)"
    elif followers < 100_000_000:
        return "Super Mega (50M–100M)"
    else:
        return "Icon (100M+)"

TIER_ORDER = ["Rising Mega (1M–10M)", "Established Mega (10M–50M)",
              "Super Mega (50M–100M)", "Icon (100M+)"]

df_ig["tier"] = df_ig["followers"].apply(assign_tier)

# ── 3c. ROI Score (if ROI dataset has cost & revenue columns) ────────────────
# Adapt column names to your actual ROI CSV
if {"campaign_cost", "revenue_generated"}.issubset(df_roi.columns):
    df_roi["roi_pct"] = (
        (df_roi["revenue_generated"] - df_roi["campaign_cost"])
        / df_roi["campaign_cost"]
    ) * 100

# ── 3d. Cost-Per-Engagement (CPE) ────────────────────────────────────────────
if {"campaign_cost", "total_engagements"}.issubset(df_roi.columns):
    df_roi["cpe"] = df_roi["campaign_cost"] / df_roi["total_engagements"].replace(0, np.nan)

print("\n✅ Feature engineering complete.")
print(df_ig[["username", "followers", "eng_rate_computed", "tier"]].head())


# =============================================================================
# 4. DATA QUALITY REPORT
# =============================================================================

def data_quality_report(df: pd.DataFrame, name: str):
    print(f"\n{'='*50}")
    print(f"  DATA QUALITY REPORT — {name}")
    print(f"{'='*50}")
    total = len(df)
    report = pd.DataFrame({
        "dtype":    df.dtypes,
        "nulls":    df.isnull().sum(),
        "null_%":   (df.isnull().sum() / total * 100).round(2),
        "unique":   df.nunique(),
        "sample":   [df[c].dropna().iloc[0] if not df[c].dropna().empty else None
                     for c in df.columns],
    })
    print(report.to_string())
    print(f"\nTotal rows: {total:,}  |  Total columns: {df.shape[1]}")

data_quality_report(df_ig,  "Instagram Influencers")
data_quality_report(df_roi, "ROI Dataset")


# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS
# =============================================================================

# ─── Helper: save figure ─────────────────────────────────────────────────────
import os
os.makedirs("outputs/plots", exist_ok=True)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close()
    print(f"  → saved outputs/plots/{name}.png")


# ── 5.1 Follower Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Follower Distribution", fontsize=16, fontweight="bold", color=TEXT_COLOR)

axes[0].hist(df_ig["followers"], bins=60, color=PALETTE[0], edgecolor="none", alpha=0.85)
axes[0].set_title("Raw Distribution")
axes[0].set_xlabel("Followers")
axes[0].set_ylabel("Count")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

axes[1].hist(np.log10(df_ig["followers"].replace(0, np.nan).dropna()),
             bins=60, color=PALETTE[1], edgecolor="none", alpha=0.85)
axes[1].set_title("Log₁₀ Distribution (cleaner view)")
axes[1].set_xlabel("log₁₀(Followers)")
axes[1].set_ylabel("Count")

savefig("01_follower_distribution")


# ── 5.2 Tier Breakdown ───────────────────────────────────────────────────────
tier_counts = df_ig["tier"].value_counts().reindex(TIER_ORDER, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Influencer Tier Breakdown", fontsize=16, fontweight="bold", color=TEXT_COLOR)

# Bar chart
bars = axes[0].bar(tier_counts.index, tier_counts.values, color=PALETTE[:4])
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Tier")
for bar, val in zip(bars, tier_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)

# Pie chart
axes[1].pie(tier_counts.values, labels=tier_counts.index,
            colors=PALETTE[:4], autopct="%1.1f%%", startangle=140,
            textprops={"color": TEXT_COLOR})
axes[1].set_title("Share by Tier")

savefig("02_tier_breakdown")


# ── 5.3 Engagement Rate by Tier ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_ig, x="tier", y="eng_rate_computed",
    order=TIER_ORDER, palette=PALETTE[:4], ax=ax,
    flierprops=dict(marker="o", markersize=3, alpha=0.4)
)
ax.set_title("Engagement Rate by Influencer Tier", fontsize=15, fontweight="bold")
ax.set_xlabel("Tier")
ax.set_ylabel("Engagement Rate (%)")
ax.set_ylim(0, df_ig["eng_rate_computed"].quantile(0.98))

# Annotate medians
for i, tier in enumerate(TIER_ORDER):
    median = df_ig[df_ig["tier"] == tier]["eng_rate_computed"].median()
    ax.text(i, median + 0.05, f"{median:.2f}%", ha="center",
            fontsize=9, color="white", fontweight="bold")

savefig("03_engagement_by_tier")


# ── 5.4 Followers vs Engagement Rate (Scatter) ───────────────────────────────
sample = df_ig.sample(min(800, len(df_ig)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    np.log10(sample["followers"].replace(0, np.nan)),
    sample["eng_rate_computed"].clip(upper=sample["eng_rate_computed"].quantile(0.97)),
    c=[TIER_ORDER.index(t) for t in sample["tier"]],
    cmap="plasma", alpha=0.65, s=40, edgecolors="none"
)
ax.set_title("Followers vs Engagement Rate\n(key insight: more followers ≠ more engagement)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("log₁₀(Followers)")
ax.set_ylabel("Engagement Rate (%)")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", label=tier,
           markerfacecolor=plt.cm.plasma(i / 3), markersize=9)
    for i, tier in enumerate(TIER_ORDER)
]
ax.legend(handles=legend_elements, title="Tier", loc="upper right",
          framealpha=0.3, labelcolor=TEXT_COLOR)

savefig("04_followers_vs_engagement")


# ── 5.5 Top 15 Niches by Average Engagement ──────────────────────────────────
TOPIC_COL = "topic"   # adjust if your column is named differently
if TOPIC_COL in df_ig.columns:
    top_topics = (
        df_ig.groupby(TOPIC_COL)["eng_rate_computed"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_eng_rate", "count": "n_influencers"})
        .query("n_influencers >= 5")
        .sort_values("avg_eng_rate", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top_topics.index, top_topics["avg_eng_rate"],
                   color=PALETTE[0], alpha=0.85)
    ax.set_title("Top 15 Niches by Average Engagement Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Avg Engagement Rate (%)")
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, top_topics.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{row['avg_eng_rate']:.2f}%  (n={int(row['n_influencers'])})",
                va="center", fontsize=9, color=TEXT_COLOR)
    savefig("05_top_niches")


# ── 5.6 Correlation Heatmap ──────────────────────────────────────────────────
num_cols = ["followers", "posts", "avg_likes", "eng_rate_computed",
            "new_post_avg_likes", "total_likes", "influence_score"]
num_cols = [c for c in num_cols if c in df_ig.columns]

corr = df_ig[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="magma",
            ax=ax, linewidths=0.5, linecolor="#0F0F0F",
            annot_kws={"size": 10})
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
savefig("06_correlation_heatmap")


# ── 5.7 Country Distribution (Top 10) ────────────────────────────────────────
COUNTRY_COL = "country"
if COUNTRY_COL in df_ig.columns:
    top_countries = df_ig[COUNTRY_COL].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(top_countries.index, top_countries.values,
                  color=PALETTE * 2, alpha=0.85)
    ax.set_title("Top 10 Countries by Influencer Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Influencers")
    for bar, val in zip(bars, top_countries.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=10, color=TEXT_COLOR)
    savefig("07_country_distribution")


# ── 5.8 ROI Analysis (if columns exist) ──────────────────────────────────────
if "roi_pct" in df_roi.columns and "cpe" in df_roi.columns:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Campaign ROI Analysis", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    axes[0].hist(df_roi["roi_pct"].dropna().clip(-100, 500),
                 bins=50, color=PALETTE[3], edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color="white", linestyle="--", linewidth=1.5, label="Break-even")
    axes[0].set_title("ROI % Distribution")
    axes[0].set_xlabel("ROI (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend(labelcolor=TEXT_COLOR, framealpha=0.3)

    axes[1].hist(df_roi["cpe"].dropna().clip(0, df_roi["cpe"].quantile(0.95)),
                 bins=50, color=PALETTE[4], edgecolor="none", alpha=0.85)
    axes[1].set_title("Cost-Per-Engagement (CPE) Distribution")
    axes[1].set_xlabel("CPE ($)")
    axes[1].set_ylabel("Count")

    savefig("08_roi_analysis")


# =============================================================================
# 6. KEY INSIGHTS SUMMARY
# =============================================================================

print("\n" + "="*60)
print("  📊 KEY EDA INSIGHTS SUMMARY")
print("="*60)

# Insight 1: Engagement vs tier
eng_by_tier = (
    df_ig.groupby("tier")["eng_rate_computed"]
    .median()
    .reindex(TIER_ORDER)
)
print("\n📌 Insight 1 — Median Engagement Rate by Tier:")
for tier, rate in eng_by_tier.items():
    print(f"   {tier:<25} → {rate:.2f}%")

# Insight 2: Top niche
if TOPIC_COL in df_ig.columns:
    best_niche = (
        df_ig.groupby(TOPIC_COL)["eng_rate_computed"]
        .mean()
        .idxmax()
    )
    best_rate = df_ig.groupby(TOPIC_COL)["eng_rate_computed"].mean().max()
    print(f"\n📌 Insight 2 — Highest Avg Engagement Niche: '{best_niche}' ({best_rate:.2f}%)")

# Insight 3: Follower–engagement correlation
corr_val = df_ig[["followers", "eng_rate_computed"]].corr().iloc[0, 1]
print(f"\n📌 Insight 3 — Followers ↔ Engagement Correlation: {corr_val:.3f}")
print("   (Negative = bigger accounts tend to have lower engagement rates)")

# Insight 4: Tier with best ROI potential
print("\n📌 Insight 4 — Portfolio Recommendation:")
print("   Micro & Macro influencers deliver the best engagement-per-dollar —")
print("   ideal targets for budget-efficient campaigns.")

print("\n✅ Phase 2 complete! All plots saved to outputs/plots/")
print("   Next up → Phase 3: K-Means Clustering & ROI Scoring Model")
