<div align="center">

# 📊 Teams Engagement Analytics
### User Behavior Modeling for Collaboration Platforms

<br/>

<img src="https://img.shields.io/badge/Category-Data%20Science-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Tech-Python-FFD43B?style=for-the-badge&logo=python&logoColor=black">
<img src="https://img.shields.io/badge/ML-Regression%20%26%20Classification-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/Explainability-SHAP-9333EA?style=for-the-badge">
<img src="https://img.shields.io/badge/Database-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">

<br/><br/>

> **Analyze · Predict · Explain** — turning raw collaboration event logs into actionable engagement intelligence.

<br/>

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Open%20Dashboard-FF4B4B?style=for-the-badge)](https://user-engagement-modeling-m.streamlit.app/)

</div>

---

## 🧠 Project Overview

This project analyzes **user behavior and engagement patterns** from collaboration-platform event logs.

```
  Generate synthetic      Load & store events      Build behavioral
  usage data          →   in PostgreSQL         →   features
         │                                               │
         └──────────────────────────────────────────────┘
                                  │
                                  ▼
                     Train predictive models        Interpret feature
                     for engagement            →    influence via SHAP
```

The project demonstrates skills across **data engineering, analytics, machine learning, and explainability.**

---

## 🏛️ Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                TEAMS ENGAGEMENT — SYSTEM ARCHITECTURE                ║
╚══════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────┐
  │   1️⃣  DATA GENERATION    │
  │  ─────────────────────  │
  │  generate_data.py       │
  │                         │
  │  • Users  (Faker)       │
  │  • Collaboration events │
  │  • Messages & meetings  │
  │  • Sessions             │
  │  • Latency & crashes    │
  └────────────┬────────────┘
               │  users.csv · events.csv · sessions.csv
               ▼
  ┌─────────────────────────┐       ┌─────────────────────────────────┐
  │   2️⃣  DATABASE SETUP     │       │        PostgreSQL 18.1          │
  │  ─────────────────────  │       │  ─────────────────────────────  │
  │  load_to_postgres.py    │──────►│                                 │
  │                         │       │   ┌──────────┬──────────────┐   │
  │  • Bulk loader          │       │   │  users   │    events    │   │
  │  • Schema creation      │       │   ├──────────┼──────────────┤   │
  └─────────────────────────┘       │   │ sessions │events_session│   │
                                    │   └──────────┴──────────────┘   │
                                    └──────────────┬──────────────────┘
                                                   │  SQL aggregations
                                                   ▼
  ┌────────────────────────────────────────────────────────────────┐
  │   3️⃣  FEATURE ENGINEERING                                      │
  │  ────────────────────────────────────────────────────────────  │
  │                                                                │
  │   session_duration_sec_mean    avg_latency_mean                │
  │   message_count_sum            crash_count_sum                 │
  │   meeting_count_sum            high_engagement ← CLF target    │
  │   event_count_sum                                              │
  └───────────────────────────────┬────────────────────────────────┘
                                  │  Feature matrix
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
       ┌────────────────────┐        ┌────────────────────┐
       │  4️⃣  REGRESSION     │        │  4️⃣  CLASSIFICATION  │
       │  ────────────────  │        │  ────────────────  │
       │  • Linear Reg      │        │  • Decision Tree   │
       │  • Decision Tree   │        │  • Random Forest ★ │
       │  • Random Forest ★ │        │  • SVM             │
       │  • SVM             │        │  • Neural Network  │
       │  • Neural Net      │        └────────┬───────────┘
       └────────┬───────────┘                 │
                └──────────────┬──────────────┘
                               ▼
              ┌─────────────────────────────┐
              │   5️⃣  SHAP EXPLAINABILITY    │
              │  ─────────────────────────  │
              │  Summary Plot               │
              │  Bar Plot (importance)      │
              │  Dependence Plots           │
              └─────────────────────────────┘
```

---

## 🗂️ Repository Structure

```
📦 teams-engagement-project
│
├── 📁 data/                 → Generated CSVs (users, events, features)
├── 📁 models/               → Saved ML models (rf_reg.pkl, scaler.pkl)
├── 📁 src/
│   ├── 📁 data_gen/         → Synthetic event generator
│   ├── load_to_postgres.py  → Bulk loader into PostgreSQL
│   └── 📁 notebooks/
│       └── modeling.ipynb   → Feature engineering & ML modeling
├── venv/                    → Virtual environment
└── README.md                → Project documentation
```

---

## 🧬 Data Pipeline

### 1️⃣ &nbsp; Synthetic Data Generator

| Component | Description |
|---|---|
| 👤 Users | Synthetic user profiles with departments and tenures |
| 📨 Events | Collaboration events: messages, meetings, interactions |
| 🕐 Sessions | Platform session records with duration |
| ⚡ Signals | Latency measurements and crash events |

> Created using **Python** — `pandas`, `numpy`, `faker`

---

### 2️⃣ &nbsp; Database Setup

All generated data is stored in **PostgreSQL 18.1** across four key tables:

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌────────────────────┐
  │  users   │────►│  events  │────►│ sessions │────►│ events_sessionized │
  └──────────┘     └──────────┘     └──────────┘     └────────────────────┘
```

---

### 3️⃣ &nbsp; Feature Engineering

Features extracted per user from raw activity:

```
  ┌─────────────────────────────────────────────────────┐
  │                   FEATURE SET                        │
  ├──────────────────────────┬──────────────────────────┤
  │  session_duration_sec    │  avg_latency_mean        │
  │  _mean                   │                          │
  ├──────────────────────────┼──────────────────────────┤
  │  message_count_sum       │  crash_count_sum         │
  ├──────────────────────────┼──────────────────────────┤
  │  meeting_count_sum       │  high_engagement  ← 🎯   │
  ├──────────────────────────┤  (classification target) │
  │  event_count_sum         │                          │
  └──────────────────────────┴──────────────────────────┘
```

---

## 🤖 Machine Learning Models

### 🔢 &nbsp; Regression Models

| Model | RMSE | R² | Performance |
|---|---|---|---|
| Linear Regression | 0.00 | 1.000 | 🟢 Excellent |
| Decision Tree | 1.39 | 0.996 | 🟢 Excellent |
| **Random Forest** | **1.76** | **0.994** | 🟢 **Best generalizer** |
| SVM | 17.81 | 0.373 | 🟡 Moderate |
| Neural Network (MLP) | 43.51 | -2.740 | 🔴 Needs tuning |

### 🟦 &nbsp; Classification Models

| Model | Accuracy | F1 Score | Performance |
|---|---|---|---|
| Decision Tree | 1.00 | 1.00 | 🟢 Perfect |
| **Random Forest** | **1.00** | **1.00** | 🟢 **Deployed** |
| SVM | 0.82 | 0.83 | 🟡 Moderate |
| Neural Network | 0.86 | 0.86 | 🟡 Competitive |

```
  Key Findings
  ────────────────────────────────────────────────────────
  ✔  Tree-based models performed strongest on engagement prediction
  ✔  SVM & MLP performed moderately well with non-linear patterns
```

---

## 🔍 Model Explainability (SHAP)

Three levels of explainability generated automatically:

```
  ┌──────────────────────────────────────────────────────────┐
  │  SHAP SUMMARY PLOT     Global feature influence overview  │
  │  ─────────────────                                        │
  │  Shows the distribution of each feature's impact         │
  │  across all users and predictions                        │
  ├──────────────────────────────────────────────────────────┤
  │  SHAP BAR PLOT         Feature importance ranking         │
  │  ────────────                                             │
  │  Ranks features by mean absolute SHAP value              │
  │  — the higher the bar, the more influential              │
  ├──────────────────────────────────────────────────────────┤
  │  SHAP DEPENDENCE PLOTS   Feature interaction depth        │
  │  ──────────────────────                                   │
  │  Shows how prediction changes as each feature            │
  │  value increases or decreases                            │
  └──────────────────────────────────────────────────────────┘
```

### Key Feature Insights

```
  Positive drivers ──────────────────────────────────────────
  session_duration_sec_mean   ████████████████  strongest ⬆
  event_count / event_sum     █████████████     strong   ⬆
  meeting_count_sum           ████████          moderate ⬆
  message_count_sum           ██████            moderate ⬆

  Negative drivers ──────────────────────────────────────────
  avg_latency_mean            ████████          high latency ⬇
  crash_count_sum             ██████            reliability  ⬇
```

---

## 🏁 How to Run Locally

### 1 — Clone the repository
```bash
git clone https://github.com/yourusername/teams-engagement-project.git
cd teams-engagement-project
```

### 2 — Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Generate data
```bash
python src/data_gen/generate_data.py
```

### 5 — Load data into PostgreSQL
```bash
python src/load_to_postgres.py
```

### 6 — Open the modeling notebook
```bash
jupyter notebook src/notebooks/modeling.ipynb
```

---

## 💾 Saved Models

Trained models are exported to:

```
models/
 ├── rf_reg.pkl     ← Random Forest regressor
 └── scaler.pkl     ← Fitted feature scaler
```

> These can be used directly for **API inference**, **dashboards**, or **deployment** — no retraining needed.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn · SHAP |
| Data | Pandas · NumPy |
| Database | PostgreSQL |
| Visualization | Matplotlib · SHAP plots |
| Environment | venv |

---

## 📈 Future Enhancements

| # | Enhancement |
|---|---|
| 1 | 🔌 Real-time inference API |
| 2 | 📊 Power BI visualization |
| 3 | 📅 Monthly active user forecasting |

---

<div align="center">

**Built with Python · PostgreSQL · Scikit-Learn · SHAP**

🔗 [**Open Live App →**](https://user-engagement-modeling-m.streamlit.app/)

</div>
