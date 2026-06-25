# 📊 Influencer Marketing Analytics
### Instagram ROI & Cluster Analysis | Senior Data Analyst Portfolio

## 🎯 Project Overview
End-to-end data analysis pipeline identifying high-ROI Instagram influencers
using K-Means clustering and a composite scoring model across 200 top accounts.

## 🔍 Key Findings
- Engagement rate drops as follower count grows — Established Mega (10M–50M) outperforms Icon-tier (100M+) by 2–3x
- 4 distinct influencer archetypes identified: Power Influencer, Engagement King, Reach Giant, Niche Player
- Engagement Kings deliver highest ROI score despite lower raw reach

## 🛠️ Tech Stack
| Layer | Tools |
|-------|-------|
| Data wrangling | Python (pandas, numpy) |
| EDA & visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn (K-Means) |
| Dashboard | Tableau Public |

## 📁 Project Structure
influencer-analytics/

├── influencer_eda.py              # Phase 2: Data cleaning & EDA (8 plots)

├── influencer_clustering.py       # Phase 3: K-Means clustering + ROI scoring

├── prepare_dashboard_data.py      # Phase 4: Dashboard data preparation

├── outputs/

│   └── data/

│       └── influencer_scored.csv  # Final scored dataset

└── Influencer_Marketing_Analytics_Report.pdf  # Executive report

## 📊 Live Dashboard
🔗 [View on Tableau Public](https://public.tableau.com/app/profile/natthaorn.lapprasitsuk/viz/InfluencerMarketingAnalytics-InstagramROIDashboard/Dashboard1)
- Example Screen <img width="1452" height="734" alt="Screenshot 2569-06-25 at 12 03 05" src="https://github.com/user-attachments/assets/dc068651-bf8d-4e98-aabc-0e5a48e6a2c9" />



## 📄 Dataset
- Source: [Top Instagram Influencers (Cleaned)](https://www.kaggle.com/datasets/surajjha101/top-instagram-influencers-data-cleaned) — Kaggle
- 200 top-ranked Instagram accounts globally

## 🚀 How to Run
```bash
# Install dependencies
pip3 install pandas numpy matplotlib seaborn plotly scikit-learn scipy

# Phase 2 — EDA
python3 influencer_eda.py

# Phase 3 — Clustering & ROI scoring
python3 influencer_clustering.py

# Phase 4 — Prepare dashboard data
python3 prepare_dashboard_data.py
```
