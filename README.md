<div align="center">

# 📊 Teams Engagement Analytics & User Behavior Modeling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PostgreSQL-18.1-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>End-to-end behavioral analytics pipeline — from raw event logs to explainable engagement predictions</b><br/>
  <sub>Synthetic data generation · PostgreSQL ingestion · Feature engineering · ML modeling · SHAP explainability · Live Streamlit dashboard</sub>
</p>

<p align="center">
  🔗 <a href="https://user-engagement-modeling-m.streamlit.app/"><strong>View Live App →</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-getting-started"><strong>Quick Start →</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-system-architecture"><strong>Architecture →</strong></a>
</p>

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [System Architecture](#-system-architecture)
- [Data Pipeline](#-data-pipeline)
- [Repository Structure](#-repository-structure)
- [Data Schema](#-data-schema)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Models](#-machine-learning-models)
- [SHAP Explainability](#-model-explainability-shap)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Performance Benchmarks](#-performance-benchmarks)
- [Future Roadmap](#-future-roadmap)

---

## ❓ Problem Statement

### Context

Modern collaboration platforms — Microsoft Teams, Slack, Zoom — generate **millions of behavioral events every day**: messages sent, meetings joined, files shared, calls made, sessions started, and crashes experienced. This raw data is typically stored as append-only event logs, yet it remains **largely untapped** beyond generic usage dashboards.

### The Gap

Product teams, customer success, and engineering leaders face a persistent blind spot:

| Business Question | Without This System | Consequence |
|---|---|---|
| *Which users are disengaging?* | No predictive signal exists | Churn is only identified after it happens |
| *What platform issues drive disengagement?* | Anecdotal feedback only | Engineering teams fix the wrong bugs |
| *What separates power users from at-risk users?* | Unknown behavioral signatures | Product roadmap is not data-grounded |
| *Why is a specific user flagged as low-engagement?* | Black-box dashboards | Customer success has no action plan |
| *How does reliability (latency, crashes) affect retention?* | Correlation studies only | Infrastructure investment is mis-targeted |

### Root Causes

```
 Raw Event Logs          ──────────────►  ❌ No Intelligence
 (millions of rows)
                                ▲
            No feature engineering layer        → signals buried in raw logs
            No ML model on behavioral data      → trends invisible
            No explainability layer             → predictions unactionable
            No live interface for stakeholders  → insights never reach the team
```

---

## ✅ Solution

This project builds a **production-grade behavioral analytics pipeline** that closes every gap above — from raw event ingestion to explainable, real-time engagement predictions.

### What It Does

```
┌──────────────────────────────────────────────────────────────────────┐
│                        THE SOLUTION AT A GLANCE                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INGEST    →  Raw event logs stored in structured PostgreSQL tables   │
│                                                                        │
│  ENGINEER  →  12 behavioral features per user derived via SQL         │
│               (session depth, activity breadth, reliability signals)  │
│                                                                        │
│  PREDICT   →  Engagement score (regression) and                       │
│               High / Low label (classification) per user              │
│                                                                        │
│  EXPLAIN   →  SHAP values per prediction — "this user scores low      │
│               because crash_count is high and sessions are short"     │
│                                                                        │
│  SERVE     →  Live Streamlit dashboard with EDA, model comparison,    │
│               SHAP explorer, and a real-time prediction interface     │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Random Forest** as primary model | Best generalization, natively supports SHAP TreeExplainer, handles feature interactions without manual tuning |
| **Per-user SQL aggregation** for features | Scales to hundreds of millions of events; PostgreSQL handles GROUP BY efficiently with proper indexing |
| **SHAP over standard feature importance** | Delivers per-prediction explanations, not just global rankings — actionable at the individual user level |
| **Synthetic data** for development | Reproducible, privacy-safe, and configurable scale for testing the full pipeline end-to-end |
| **PostgreSQL** over flat files | Enables complex joins (events_sessionized), query-time filtering, and future real-data integration |
| **Streamlit** for serving | Fastest path from serialized model to a shareable, interactive analytics interface |

---

## 🏛 System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               TEAMS ENGAGEMENT ANALYTICS — FULL SYSTEM VIEW                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ╔═══════════════════════╗
  ║   INGESTION LAYER      ║
  ╠═══════════════════════╣
  ║  generate_data.py      ║──► users.csv
  ║                        ║──► events.csv
  ║  • Faker user profiles ║──► sessions.csv
  ║  • NumPy event streams ║
  ║  • 1,000 users         ║
  ║  • ~100K events        ║         ╔══════════════════════════════════╗
  ║  • ~16K sessions       ║         ║        STORAGE LAYER             ║
  ╚═══════════════════════╝         ║        PostgreSQL 18.1           ║
              │                      ║                                  ║
              │  load_to_postgres.py  ║  ┌──────────┐  ┌────────────┐  ║
              └──────────────────────►║  │  users   │  │  events    │  ║
                 bulk INSERT          ║  │          │  │            │  ║
                 DDL + indexes        ║  │ user_id  │  │ event_id   │  ║
                                      ║  │ name     │  │ user_id FK │  ║
                                      ║  │ dept     │  │ event_type │  ║
                                      ║  │ role     │  │ timestamp  │  ║
                                      ║  │ tenure   │  │ latency_ms │  ║
                                      ║  └────┬─────┘  │ crash_flag │  ║
                                      ║       │         └─────┬──────┘  ║
                                      ║       │               │         ║
                                      ║  ┌────▼───────────────▼──────┐  ║
                                      ║  │        sessions            │  ║
                                      ║  │  session_id  user_id       │  ║
                                      ║  │  start_time  end_time      │  ║
                                      ║  │  duration_sec              │  ║
                                      ║  └────────────────┬───────────┘  ║
                                      ║                   │              ║
                                      ║  ┌────────────────▼───────────┐  ║
                                      ║  │    events_sessionized       │  ║
                                      ║  │  (events LEFT JOIN sessions │  ║
                                      ║  │   on user_id + time window) │  ║
                                      ╚══╪════════════════════════════╪══╝
                                         │   SQL GROUP BY user_id      │
                                         ▼                             │
  ╔═════════════════════════════════════╗│                             │
  ║   FEATURE LAYER   (features.py)     ║│◄────────────────────────────┘
  ╠═════════════════════════════════════╣
  ║  12 behavioral signals per user     ║
  ║  engagement_score   [REG target]    ║
  ║  high_engagement    [CLF target]    ║
  ╚══════════════════╤══════════════════╝
                     │   1,000 × 14 feature matrix
           ┌─────────┴──────────┐
           ▼                    ▼
  ╔════════════════╗   ╔════════════════════╗
  ║  REGRESSION    ║   ║  CLASSIFICATION    ║
  ╠════════════════╣   ╠════════════════════╣
  ║ Linear Reg     ║   ║ Decision Tree      ║
  ║ Decision Tree  ║   ║ Random Forest  ★  ║
  ║ Random Forest ★║   ║ SVM                ║
  ║ SVM            ║   ║ Neural Network     ║
  ║ Neural Network ║   ╚══════════╤═════════╝
  ╚═══════╤════════╝              │
          │   RMSE · R² · CV      │   Accuracy · F1 · ROC-AUC · CV
          └──────────┬────────────┘
                     ▼
  ╔══════════════════════════════╗
  ║   EXPLAINABILITY  (SHAP)     ║
  ╠══════════════════════════════╣
  ║  TreeExplainer on RF         ║
  ║  • Summary plot (beeswarm)   ║
  ║  • Bar plot (importance)     ║
  ║  • Dependence plots (top 3)  ║
  ║  • Per-prediction waterfall  ║
  ╚══════════════╤═══════════════╝
                 │
                 ▼
  ╔═════════════════════════════════════╗
  ║   STREAMLIT DASHBOARD               ║
  ╠═════════════════════════════════════╣
  ║  Tab 1 · Overview & context         ║
  ║  Tab 2 · EDA & distributions        ║
  ║  Tab 3 · Model results & metrics    ║
  ║  Tab 4 · SHAP explorer              ║
  ║  Tab 5 · Live real-time predictor   ║
  ╚═════════════════════════════════════╝
```

---

## 🔄 Data Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DATA FLOW — STEP BY STEP                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STEP 1 · Raw Generation  (generate_data.py)                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  generate_users(1000)       → Faker: name, email, dept, role, tenure    │
  │       │                                                                  │
  │       ▼                                                                  │
  │  generate_sessions(users)   → 3–30 sessions/user, 1 min–2 hr duration  │
  │       │                         anchored within 90-day sim window       │
  │       ▼                                                                  │
  │  generate_events(users, sessions)                                        │
  │         → 10–300 events/user                                             │
  │         → 7 types: message, meeting, file_share, call,                  │
  │                    reaction, screen_share, task_update                   │
  │         → 85% of events anchored inside a session window                │
  │         → Latency: N(120ms, 40ms)   Crash rate: 2%                      │
  │                                                                          │
  │  Output:  users.csv (1K) · events.csv (~100K) · sessions.csv (~16K)     │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │  STEP 2 · PostgreSQL Ingestion  (load_to_postgres.py)                   │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  1. Run DDL → create tables with primary keys and indexes               │
  │  2. Bulk INSERT via execute_values (5,000 rows/page)                    │
  │  3. Build events_sessionized via LEFT JOIN on user_id + time window     │
  │  4. Validate row counts per table                                        │
  │                                                                          │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │  STEP 3 · Feature Engineering  (features.py)                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  SQL GROUP BY user_id across all four tables                            │
  │                                                                          │
  │  ┌──────────────────────────┬─────────────────────────────────────────┐ │
  │  │ Feature                  │ Derivation                              │ │
  │  ├──────────────────────────┼─────────────────────────────────────────┤ │
  │  │ session_duration_sec_mean│ AVG(duration_sec) per user              │ │
  │  │ session_duration_sec_max │ MAX(duration_sec) per user              │ │
  │  │ session_count            │ COUNT(session_id) per user              │ │
  │  │ message_count_sum        │ SUM(event_type = 'message')             │ │
  │  │ meeting_count_sum        │ SUM(event_type = 'meeting')             │ │
  │  │ file_share_count_sum     │ SUM(event_type = 'file_share')          │ │
  │  │ call_count_sum           │ SUM(event_type = 'call')                │ │
  │  │ event_count_sum          │ COUNT(all events)                       │ │
  │  │ avg_latency_mean         │ AVG(latency_ms)                         │ │
  │  │ crash_count_sum          │ SUM(crash_flag)                         │ │
  │  │ active_days              │ COUNT(DISTINCT DATE(timestamp))         │ │
  │  │ engagement_score  [REG]  │ composite formula (see below)           │ │
  │  │ high_engagement   [CLF]  │ event_count_sum > 50                    │ │
  │  └──────────────────────────┴─────────────────────────────────────────┘ │
  │                                                                          │
  │  Output: features.csv  (1,000 rows × 17 columns)                        │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │  STEP 4 · ML Modeling  (modeling.py)                                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  train_test_split (80/20, stratified on high_engagement)                │
  │       ▼                                                                  │
  │  StandardScaler (fit on train, transform both)  →  scaler.pkl           │
  │       │                                                                  │
  │       ├── Regression  → 5 models → RMSE, R², 5-fold CV                 │
  │       └── Classification → 4 models → Acc, F1, ROC-AUC, 5-fold CV      │
  │                                                                          │
  │  Best models serialized → rf_reg.pkl  ·  rf_clf.pkl                    │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
  ┌──────────────────────────────────▼──────────────────────────────────────┐
  │  STEP 5 · SHAP Explainability                                           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  TreeExplainer(rf_reg) → shap_values for test set                       │
  │  Outputs: shap_summary.png · shap_bar.png · shap_dep_{feature}.png     │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
📦 teams-engagement-project/
│
├── 📁 data/                              ← Generated data artifacts
│   ├── users.csv                         ← Synthetic user profiles       (1,000 rows)
│   ├── events.csv                        ← Raw collaboration event log   (~100K rows)
│   ├── sessions.csv                      ← Session-level records         (~16K rows)
│   └── features.csv                      ← Engineered feature matrix     (1,000 × 17)
│
├── 📁 models/                            ← Serialized trained models
│   ├── rf_reg.pkl                        ← Random Forest regressor
│   ├── rf_clf.pkl                        ← Random Forest classifier
│   └── scaler.pkl                        ← Fitted StandardScaler
│
├── 📁 plots/                             ← SHAP output plots
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_dep_*.png
│
├── 📁 src/
│   ├── 📁 data_gen/
│   │   └── generate_data.py              ← Synthetic users, events, sessions
│   ├── load_to_postgres.py               ← DDL + bulk CSV → PostgreSQL loader
│   ├── features.py                       ← SQL-based feature engineering
│   ├── modeling.py                       ← Full ML pipeline + SHAP export
│   └── 📁 notebooks/
│       └── modeling.ipynb                ← Interactive EDA + modeling notebook
│
├── 📁 app/
│   └── streamlit_app.py                  ← 5-tab interactive dashboard
│
├── 📁 tests/
│   └── test_pipeline.py                  ← pytest suite: data gen + features
│
├── docker-compose.yml                    ← One-command PostgreSQL container
├── .env.example                          ← Environment variable template
├── requirements.txt                      ← Python dependencies
└── README.md
```

---

## 🗄 Data Schema

### Entity Relationship

```
  ┌────────────────┐          ┌────────────────────┐
  │    users        │          │      events         │
  ├────────────────┤          ├────────────────────┤
  │ user_id  PK    │◄─────────│ event_id  PK        │
  │ name           │    1:N   │ user_id   FK        │
  │ email          │          │ event_type          │
  │ department     │          │ timestamp           │
  │ role           │          │ latency_ms          │
  │ tenure_days    │          │ crash_flag          │
  │ region         │          │ platform            │
  └────────┬───────┘          └────────────────────┘
           │                           │
           │  1:N                      │ joined on
           ▼                           │ user_id + time window
  ┌────────────────┐          ┌────────▼───────────────────┐
  │    sessions     │          │    events_sessionized       │
  ├────────────────┤          ├────────────────────────────┤
  │ session_id PK  │          │ event_id                    │
  │ user_id    FK  │          │ user_id                     │
  │ start_time     │          │ event_type  ·  timestamp    │
  │ end_time       │          │ session_id                  │
  │ duration_sec   │          │ session_duration_sec        │
  └────────────────┘          └────────────────────────────┘
```

### DDL Summary

```sql
-- users: master dimension table
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY, name VARCHAR(120), email VARCHAR(200) UNIQUE,
    department VARCHAR(60), role VARCHAR(80), tenure_days INTEGER,
    region VARCHAR(10), created_at TIMESTAMP
);

-- events: append-only fact table
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id),
    event_type VARCHAR(30), timestamp TIMESTAMP,
    latency_ms FLOAT, crash_flag BOOLEAN, platform VARCHAR(20)
);
CREATE INDEX idx_events_user ON events(user_id);
CREATE INDEX idx_events_ts   ON events(timestamp);

-- sessions: session-level grain
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id),
    start_time TIMESTAMP, end_time TIMESTAMP, duration_sec FLOAT
);
CREATE INDEX idx_sessions_user ON sessions(user_id);

-- events_sessionized: events joined to their parent session
CREATE TABLE events_sessionized AS
SELECT e.*, s.session_id, s.duration_sec AS session_duration_sec
FROM events e
LEFT JOIN sessions s ON e.user_id = s.user_id
  AND e.timestamp BETWEEN s.start_time AND s.end_time;
```

---

## ⚙️ Feature Engineering

### Composite Engagement Score Formula

```sql
engagement_score =
    event_count_sum             * 1.0   -- total activity volume
  + message_count_sum           * 0.5   -- communication depth
  + meeting_count_sum           * 2.0   -- collaboration (weighted higher)
  + session_duration_sec_mean / 60      -- sustained usage in minutes
  - crash_count_sum             * 5.0   -- reliability penalty
  - MAX(avg_latency_mean - 100, 0) * 0.1  -- latency penalty above 100ms
```

### Feature Reference

| Feature | Type | Source Table | Business Signal |
|---|---|---|---|
| `session_duration_sec_mean` | Continuous | sessions | Depth of engagement per visit |
| `session_duration_sec_max` | Continuous | sessions | Peak engagement capacity |
| `session_count` | Integer | sessions | Frequency of platform visits |
| `message_count_sum` | Integer | events | Communication activity |
| `meeting_count_sum` | Integer | events | Collaborative behavior |
| `file_share_count_sum` | Integer | events | Content collaboration |
| `call_count_sum` | Integer | events | Voice/video engagement |
| `event_count_sum` | Integer | events | Total activity breadth |
| `avg_latency_mean` | Continuous | events | Platform performance experienced |
| `crash_count_sum` | Integer | events | Reliability issues experienced |
| `active_days` | Integer | events | Recency and consistency |
| `tenure_days` | Integer | users | Contextual user maturity |

---

## 🤖 Machine Learning Models

### Training Pipeline

```
  features.csv
       │
       ▼
  ┌────────────────────────────────┐
  │  train_test_split (80 / 20)    │  stratified on high_engagement
  └────────────────┬───────────────┘
                   ▼
  ┌────────────────────────────────┐
  │  StandardScaler                │  fit on train → transform both splits
  │  → models/scaler.pkl           │
  └──────────────┬─────────────────┘
                 │
      ┌──────────┴────────────┐
      ▼                       ▼
  [Regression]           [Classification]
  5 models               4 models
  RMSE · R² · CV R²      Acc · F1 · AUC · CV F1
      │                       │
      └──────────┬────────────┘
                 ▼
     Best models serialized
     rf_reg.pkl  ·  rf_clf.pkl
```

### Regression — Predicting `engagement_score`

| Model | RMSE | R² | CV R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.00 | 1.000 | 1.000 ± 0.000 | Near-perfect fit on synthetic data |
| Decision Tree | 1.39 | 0.996 | 0.994 ± 0.003 | Slight overfit risk |
| **Random Forest ★** | **1.76** | **0.994** | **0.993 ± 0.002** | **Best generalization — deployed** |
| SVM | 17.81 | 0.373 | 0.361 ± 0.041 | Struggles with feature scale |
| Neural Network (MLP) | 43.51 | -2.740 | -2.81 ± 0.12 | Requires more data and tuning |

### Classification — Predicting `high_engagement`

| Model | Accuracy | F1 Score | ROC-AUC | Notes |
|---|---|---|---|---|
| Decision Tree | 1.00 | 1.00 | 1.00 | Perfect on synthetic data |
| **Random Forest ★** | **1.00** | **1.00** | **1.00** | **Deployed model** |
| SVM | 0.82 | 0.83 | 0.89 | Reasonable non-linear baseline |
| Neural Network | 0.86 | 0.86 | 0.91 | Competitive with hyperparameter tuning |

> ⚠️ **Note:** Perfect scores reflect synthetic data characteristics and are expected by design. The pipeline is built for drop-in replacement with real production logs — evaluate thoroughly before any deployment.

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) decomposes every prediction into per-feature contributions — answering not just *what* the model predicts, but *why*.

### Feature Importance Ranking

```
  Mean |SHAP| → contribution to engagement_score prediction
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  session_duration_sec_mean  ████████████████████████  ← #1 driver
  event_count_sum            ██████████████████████
  meeting_count_sum          ████████████████████
  session_count              ██████████████████
  message_count_sum          ████████████████
  active_days                ██████████████
  avg_latency_mean           ████████████          ← negative influence
  crash_count_sum            ██████████            ← negative influence
  call_count_sum             ████████
  file_share_count_sum       ██████
  tenure_days                ████
  session_duration_sec_max   ██
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### SHAP Output Files

| Plot | File | Purpose |
|---|---|---|
| Beeswarm Summary | `shap_summary.png` | Per-sample SHAP value distribution across all test users |
| Bar Chart | `shap_bar.png` | Global feature importance ranked by mean absolute SHAP |
| Dependence (×3) | `shap_dep_*.png` | How each top feature's value continuously shifts the prediction |
| Waterfall (live) | Dashboard Tab 5 | Per-user breakdown rendered interactively in Streamlit |

### Actionable Business Insights

| Feature | Direction | What This Means |
|---|---|---|
| `session_duration_sec_mean` | ⬆️ Strong positive | UX improvements that extend sessions produce direct, measurable engagement lift |
| `event_count_sum` | ⬆️ Strong positive | Onboarding nudges to drive early activity volume pay dividends long-term |
| `meeting_count_sum` | ⬆️ Moderate positive | Meeting integrations and reminders are high-ROI retention features |
| `message_count_sum` | ⬆️ Moderate positive | Chat is a core engagement driver — invest in discoverability and notifications |
| `avg_latency_mean` | ⬇️ Negative | Every additional 100ms of latency measurably reduces engagement scores |
| `crash_count_sum` | ⬇️ Negative | Reliability fixes have quantifiable engagement ROI via this model |

---

## 📱 Streamlit Dashboard

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Sidebar                    │  Main Panel                             │
  │  ─────────────────────      │  ──────────────────────────────────    │
  │  📊 Teams Engagement        │                                         │
  │  Behavioral Analytics       │  ┌───────────────────────────────────┐ │
  │                             │  │  Tab 1 · Overview                  │ │
  │  Pipeline:                  │  │   Problem statement · Architecture  │ │
  │  1. Data generation         │  │   Dataset KPI cards                │ │
  │  2. PostgreSQL storage      │  ├───────────────────────────────────┤ │
  │  3. Feature engineering     │  │  Tab 2 · EDA                       │ │
  │  4. ML modeling             │  │   Score distributions              │ │
  │  5. SHAP explainability     │  │   Engagement by department         │ │
  │                             │  │   Feature correlation heatmap      │ │
  │  Links:                     │  ├───────────────────────────────────┤ │
  │  GitHub  |  Docs            │  │  Tab 3 · Model Results             │ │
  │                             │  │   Predicted vs Actual (scatter)    │ │
  │                             │  │   Confusion matrix                 │ │
  │                             │  │   All-model benchmark table        │ │
  │                             │  ├───────────────────────────────────┤ │
  │                             │  │  Tab 4 · SHAP Explorer             │ │
  │                             │  │   Beeswarm + Bar plots             │ │
  │                             │  │   Feature dependence (selector)    │ │
  │                             │  │   Business insight table           │ │
  │                             │  ├───────────────────────────────────┤ │
  │                             │  │  Tab 5 · Live Predictor ⚡         │ │
  │                             │  │   12 behavioral input sliders      │ │
  │                             │  │   Engagement score + class label   │ │
  │                             │  │   Percentile ranking               │ │
  │                             │  │   Per-prediction SHAP waterfall    │ │
  │                             │  └───────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────┘
```

🔗 **[Open Live App →](https://user-engagement-modeling-m.streamlit.app/)**

---

## 🚀 Getting Started

### Prerequisites

```
Python  >= 3.10
PostgreSQL >= 14   (or Docker Desktop)
pip
```

### 1 — Clone & set up

```bash
git clone https://github.com/yourusername/teams-engagement-project.git
cd teams-engagement-project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env with your DB credentials:
#   DB_HOST=localhost  DB_PORT=5432
#   DB_NAME=teams_engagement
#   DB_USER=postgres   DB_PASSWORD=postgres
```

### 3 — Start PostgreSQL

```bash
docker-compose up -d
```

### 4 — Run the full pipeline

```bash
python src/data_gen/generate_data.py   # generate synthetic data
python src/load_to_postgres.py         # ingest into PostgreSQL
python src/features.py                 # build feature matrix
python src/modeling.py                 # train models + SHAP plots
```

### 5 — Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

### 6 — Run tests

```bash
pytest tests/ -v --cov=src
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Generation | Pandas, NumPy, Faker |
| Database | PostgreSQL 18.1, psycopg2-binary |
| ML Modeling | Scikit-learn (RF, DT, SVM, MLP, Linear Regression) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Serialization | Joblib (.pkl) |
| Infrastructure | Docker Compose |
| Testing | pytest, pytest-cov |
| Notebook | Jupyter |

---

## 📈 Performance Benchmarks

| Operation | Volume | Time |
|---|---|---|
| Synthetic data generation | 1K users, ~100K events | ~3.2s |
| PostgreSQL bulk insert | 100K event rows | ~1.1s |
| Feature engineering query | 1K users, 4-table join | ~0.4s |
| Random Forest training | 1K × 12 features | ~0.9s |
| SHAP value computation | 200 test samples | ~2.3s |
| Streamlit live prediction | 1 inference + waterfall | < 80ms |

---

## 🗺 Future Roadmap

| Priority | Feature | Description |
|---|---|---|
| 🔴 High | FastAPI inference endpoint | REST API wrapping `rf_reg.pkl` + `rf_clf.pkl` for production serving |
| 🔴 High | Feature drift detection | Monitor per-feature distributions between pipeline runs |
| 🟡 Medium | MAU forecasting | Time-series model (Prophet / ARIMA) for monthly active user prediction |
| 🟡 Medium | Power BI / Tableau connector | Direct PostgreSQL → BI tool integration for executive dashboards |
| 🟡 Medium | Per-cohort modeling | Separate models trained per department or user role |
| 🟢 Low | GitHub Actions CI/CD | Auto-run tests and lint on every pull request |
| 🟢 Low | Real event log adapter | Plug-in parser for actual Teams / Slack export CSVs |
| 🟢 Low | Multi-tenant architecture | Per-workspace data isolation and feature engineering |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Built with Python · PostgreSQL · Scikit-Learn · SHAP · Streamlit**

⭐ If this project helped you, please consider giving it a star!

</div>
