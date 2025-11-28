## 📊 Teams Engagement Analytics \& User Behavior Modeling

<img src="https://img.shields.io/badge/Category-Data%20Science-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/Tech-Python-yellow?style=for-the-badge"> <img src="https://img.shields.io/badge/ML-Regression%20%26%20Classification-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Explainability-SHAP-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/Database-PostgreSQL-lightblue?style=for-the-badge">

🔗 Live App:
https://user-engagement-modeling-m.streamlit.app/

🧠 Project Overview



This project analyzes user behavior and engagement patterns from collaboration-platform event logs.

The goal is to:



generate scalable synthetic usage data



load and store events efficiently in PostgreSQL



build behavioral features



train predictive models for engagement



interpret feature influence using SHAP



The project demonstrates skills in data engineering, analytics, machine learning, and explainability.



🗂️ Repository Structure

📦 teams-engagement-project

├── 📁 data/                 → Generated CSVs (users, events, features)

├── 📁 models/               → Saved ML models (rf\_reg.pkl, scaler.pkl)

├── 📁 src/

│   ├── 📁 data\_gen/         → Synthetic event generator

│   ├── load\_to\_postgres.py → Bulk loader into PostgreSQL

│   └── 📁 notebooks/

│       └── modeling.ipynb  → Feature engineering \& ML modeling

├── venv/                    → Virtual environment

└── README.md                → Project documentation



🧬 Data Pipeline

1️⃣ Synthetic Data Generator



✔ Users

✔ Collaboration events

✔ Messages, meetings, sessions

✔ Latency, crashes, interactions



Created using Python (pandas, numpy, faker).



2️⃣ Database Setup



All generated data is stored in PostgreSQL 18.1.



Key tables:



users



events



sessions



events\_sessionized



3️⃣ Feature Engineering



Features extracted from user activity:



session\_duration\_sec\_mean



message\_count\_sum



meeting\_count\_sum



event\_count\_sum



avg\_latency\_mean



crash\_count\_sum



high\_engagement (classification target)



🤖 Machine Learning Models

🔢 Regression Models

Model	RMSE	R²

Linear Regression	0.00	1.000

Decision Tree	1.39	0.996

Random Forest	1.76	0.994

SVM	17.81	0.373

Neural Network (MLP)	43.51	-2.740

🟦 Classification Models

Model	Accuracy	F1 Score

Decision Tree	1.00	1.00

Random Forest	1.00	1.00

SVM	0.82	0.83

Neural Network	0.86	0.86



✔ Tree-based models performed the strongest on engagement prediction.

✔ SVM \& MLP performed moderately well with non-linear patterns.



🔍 Model Explainability (SHAP)

✔ SHAP Summary Plot



Shows global influence of each feature on engagement prediction.



✔ SHAP Bar Plot



Ranks the most impactful features.



✔ SHAP Dependence Plots



Shows how interaction intensity changes prediction values.



Key Feature Insights



session\_duration\_sec\_mean → strongest positive influence



event\_count\_mean / event\_count\_sum → strong activity indicators



meeting\_count\_sum → contributes to consistent engagement



message\_count\_sum → communication-driven interaction



avg\_latency\_mean → higher latency reduces engagement



crash\_count\_sum → reliability impacts behavior



🏁 How to Run the Project Locally

1\. Clone the repository

git clone https://github.com/yourusername/teams-engagement-project.git

cd teams-engagement-project



2\. Create a virtual environment

python -m venv venv

venv\\Scripts\\activate



3\. Install dependencies

pip install -r requirements.txt



4\. Generate data

python src/data\_gen/generate\_data.py



5\. Load data into PostgreSQL

python src/load\_to\_postgres.py



6\. Open the modeling notebook

jupyter notebook src/notebooks/modeling.ipynb



💾 Saved Models



Models are exported to:



models/

 └── rf\_reg.pkl

 └── scaler.pkl





These can be used for API inference, dashboards, or deployment.



🛠️ Tech Stack

Area	Tools

Programming	Python

ML	sklearn, SHAP

Data	Pandas, NumPy

Database	PostgreSQL

Visualization	Matplotlib, SHAP

Environment	venv

📈 Future Enhancements




Real-time inference API



Power BI visualization



Monthly active user forecasting




