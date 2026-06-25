# =============================================================================
# INFLUENCER MARKETING ANALYTICS — Phase 4: Tableau Public Dashboard
# Complete Step-by-Step Guide for Apple Silicon Mac
# =============================================================================

# =============================================================================
# STEP 0 — INSTALL TABLEAU PUBLIC ON APPLE SILICON MAC
# =============================================================================

# 1. Go to: https://public.tableau.com/app/discover
# 2. Click "Download Tableau Public"
# 3. Download the .dmg file (it is Apple Silicon native as of 2023.1+)
# 4. Open the .dmg → drag Tableau Public to Applications
# 5. Open Tableau Public → sign in with a free account

# NOTE: If you see "cannot be opened because Apple cannot check it for malware"
#       → Go to System Settings → Privacy & Security → click "Open Anyway"


# =============================================================================
# STEP 1 — PREPARE YOUR DATA FILE
# =============================================================================

# Your dashboard data source = outputs/data/influencer_scored.csv
# Run this Python script first to make sure it's clean and enriched:

import pandas as pd
import numpy as np
import os

os.makedirs("outputs/data", exist_ok=True)

PATH = "outputs/data/influencer_scored.csv"

df = pd.read_csv(PATH)

# ── Add extra columns Tableau will need ──────────────────────────────────────

# 1. Follower range label (for filters)
def follower_range(f):
    if f < 10_000_000:    return "1M–10M"
    elif f < 50_000_000:  return "10M–50M"
    elif f < 100_000_000: return "50M–100M"
    else:                 return "100M+"

df["follower_range"] = df["followers"].apply(follower_range)

# 2. Engagement tier label
def eng_tier(e):
    if e >= 5:    return "High (5%+)"
    elif e >= 2:  return "Medium (2–5%)"
    else:         return "Low (<2%)"

df["engagement_tier"] = df["eng_rate_computed"].apply(eng_tier)

# 3. Followers in millions (cleaner Tableau labels)
df["followers_M"] = (df["followers"] / 1_000_000).round(2)

# 4. Clean score grade (remove emoji for Tableau compatibility)
grade_clean = {
    "🥇 Tier A — Premium":  "Tier A — Premium",
    "🥈 Tier B — Strong":   "Tier B — Strong",
    "🥉 Tier C — Moderate": "Tier C — Moderate",
    "⚪ Tier D — Low":      "Tier D — Low",
}
if "score_grade" in df.columns:
    df["score_grade_clean"] = df["score_grade"].map(grade_clean).fillna(df["score_grade"])

# 5. Rank column
df["roi_rank"] = df["roi_score"].rank(ascending=False).astype(int)

# Save enriched file
df.to_csv("outputs/data/influencer_dashboard_data.csv", index=False)

print("✅ Dashboard data saved → outputs/data/influencer_dashboard_data.csv")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {list(df.columns)}")
print("\n📋 Column Reference for Tableau:")
for col in df.columns:
    dtype = str(df[col].dtype)
    sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
    print(f"   {col:<30} {dtype:<12} e.g. {sample[:30]}")
