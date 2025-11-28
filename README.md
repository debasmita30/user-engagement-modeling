## User Engagement Modeling

A full analytical pipeline that generates synthetic user activity data, sessionizes event streams, engineers behavioral features, trains machine learning models, and serves real-time predictions through an interactive application.

🔗 Live App:
https://diya-engagement.streamlit.app

This repository demonstrates strong capability in data engineering, feature engineering, machine learning, and dashboard development—aligned with real-world workplace engagement analytics.

1. Project Overview

This project simulates user behavior in a collaboration environment and builds a machine learning model to estimate engagement levels based on event logs and usage metrics.

The workflow contains:

Synthetic data generation (messages, meetings, sessions)

Event sessionization using inactivity thresholds

Feature engineering across multiple activity dimensions

ML model training with Random Forest

Model interpretability through feature importance

Deployment-ready Streamlit interface

Optional PostgreSQL integration

2. Key Features

Data Pipeline

User and event generators written in Python

Session stitching logic (30-minute inactivity rule)

Aggregated behavior metrics per user

PostgreSQL-compatible data outputs

Modeling & Feature Engineering

feature_engineering.ipynb notebook performs:

Feature cleaning

Complex aggregation

Distribution analysis

Outlier review

Final dataset preparation

Multiple ML models trained & compared

Random Forest chosen as primary predictor

Scaler + model saved for inference

Application Layer

Streamlit UI for:

Single-record prediction

Feature importance charts

Local explanation visualization

Live PostgreSQL queries

Exportable prediction report

3. Repository Structure
   
user-engagement-modeling/
│
├── data/                      
│   ├── users.csv
│   ├── events.csv
│   ├── features.csv
│   └── features.parquet
│


├── models/                   
│   ├── rf_reg.pkl
│   ├── scaler.pkl
│   └── feature_means.json
│


├── src/
│   ├── data_gen/
│   │   └── generate_data.py
│   ├── load_to_postgres.py
│   └── notebooks/
│       ├── modeling.ipynb              # ML model training
│       └── feature_engineering.ipynb   # Data → Features pipeline
│


├── streamlit_app.py          
├── requirements.txt
└── README.md

5. How to Run Locally
   
Environment Setup

python -m venv venv

venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Start Streamlit Application

streamlit run streamlit_app.py

5. Data Pipeline Summary
   
1. Synthetic Generation

Creates:

Users

Meetings

Messages

Latency readings

Crash logs

Session timestamps

2. Sessionization Logic

Events are grouped into sessions using:

30-minute inactivity gap

Per-user event ordering

End timestamps + session durations

3. Feature Engineering Notebook

feature_engineering.ipynb produces:

Session statistics

Activity intensity measures

Event/meeting/message aggregates

Crash behavior

Latency averages

Final features.csv + features.parquet

4. Machine Learning

modeling.ipynb trains:

Linear Regression

Decision Tree

Random Forest (chosen)

SVM

MLP Neural Network

Artifacts saved for deployment:

rf_reg.pkl

scaler.pkl

feature_means.json

6. Streamlit Application Overview
   
Prediction Interface

Users enter activity metrics

Input is standardized using stored scaler

Random Forest model predicts engagement score

Interpretability

Global feature importance

Local contribution analysis

Operational Features

PostgreSQL live query support

Report download option

🔗 Live App:
https://diya-engagement.streamlit.app

7. Technology Stack

Python 3.10+

Pandas, NumPy

Scikit-learn

Matplotlib

Joblib

Streamlit

PostgreSQL (optional)

JupyterLab for analysis

8. Ideal Use Cases

This project applies to:

Employee engagement analytics

Workspace behavior modeling

Product usage scoring

Productivity platforms

Early churn detection

User insights for product teams
