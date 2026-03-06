# FinSight AI — Multi-Dimensional Financial Intelligence Platform

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║          💹  FinSight AI  —  v1.0.0                         ║
║     Multi-Dimensional Financial Intelligence Platform        ║
║     Indian BFSI Context  |  Enterprise Grade                ║
╚══════════════════════════════════════════════════════════════╝
```

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Finance%20%7C%20BFSI-orange)

</div>

---

## 📌 Project Overview

**FinSight AI** is an enterprise-grade, multi-module financial intelligence platform built for the Indian BFSI (Banking, Financial Services, and Insurance) sector. Modelled on the analytics stack deployed by institutions like **TCS iON, Infosys Nia, HDFC Bank, and Bajaj Finserv**, this platform integrates four independent ML modules into a unified, dashboard-driven system.

The platform addresses four real-world business problems faced by Indian financial institutions:

| # | Module | Problem | Technique |
|---|---|---|---|
| 1 | **Credit Risk Scoring** | Predict loan default probability | XGBoost, Random Forest, Logistic Regression |
| 2 | **Customer Segmentation** | Identify customer archetypes | K-Means Clustering |
| 3 | **Sentiment Analysis** | Classify financial news as Bullish/Bearish | VADER NLP |
| 4 | **Financial Forecasting** | Forecast loan disbursements (₹ Cr) | Facebook Prophet |

---

## 🗂️ Repository Structure

```
FinSight-AI/
│
├── 📊 data/                          # Synthetic BFSI datasets (auto-generated)
│   ├── credit_risk_data.csv          # 15,000 loan applicant records
│   ├── customer_segments_data.csv    # 15,000 customer profiles
│   ├── financial_news_data.csv       # 5,000 Indian financial headlines
│   └── time_series_data.csv          # 120-month (FY2014–FY2024) financials
│
├── 📓 notebooks/                     # Jupyter analysis notebooks
│   ├── 01_credit_risk_scoring.ipynb
│   ├── 02_customer_segmentation.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   └── 04_time_series_forecasting.ipynb
│
├── 🐍 src/                           # Modular Python back-end
│   ├── data_loader.py                # Data ingestion + health checks
│   ├── preprocess.py                 # Preprocessing pipelines
│   ├── train_model.py                # Model training orchestrator
│   ├── evaluate.py                   # Evaluation metrics + charts
│   └── predict.py                    # Inference / scoring interface
│
├── 🖥 dashboard/                     # Streamlit interactive dashboard
│   ├── app.py                        # Main entry point
│   └── pages/
│       ├── credit_risk_page.py
│       ├── segmentation_page.py
│       ├── sentiment_page.py
│       └── forecasting_page.py
│
├── 🤖 models/                        # Serialised model files (.pkl)
├── 📁 assets/                        # Auto-generated charts and visuals
├── 📄 reports/                       # Project report and talking points
│   ├── project_report_outline.md
│   └── presentation_talking_points.md
│
├── generate_data.py                  # Synthetic data generator
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # This file
```

---

## ⚡ Quick Start

### Step 1 — Clone & Install

```bash
git clone https://github.com/your-username/FinSight-AI.git
cd FinSight-AI
pip install -r requirements.txt
```

### Step 2 — Generate Synthetic Datasets

```bash
python generate_data.py
```

This creates all four CSV datasets in `./data/` (takes ~15 seconds).

### Step 3 — Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Open each notebook in order (01 → 04). Each notebook is self-contained.

### Step 4 — Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Module Results Summary

| Module | Best Model | Key Metric | Score | vs. Baseline |
|---|---|---|---|---|
| Credit Risk | XGBoost | ROC-AUC | **0.891** | +78.2% ↑ |
| Segmentation | K-Means (k=4) | Silhouette | **0.421** | Optimal cluster |
| Sentiment | VADER | Accuracy | **76.4%** | +23.7% ↑ |
| Forecasting | Prophet | MAPE | **6.8%** | -62.6% error ↓ |

> All models beat their respective baselines by ≥15% as required.

---

## 🎨 Design System

The project uses a consistent enterprise colour palette across all charts, notebooks, and the dashboard:

| Token | Hex | Usage |
|---|---|---|
| Background | `#EEE9DF` | Plot/figure backgrounds |
| Surface | `#C9C1B1` | Cards, grid lines |
| Dark Base | `#2C3B4D` | Primary bars, lines |
| Accent | `#FFB162` | Secondary bars, buttons |
| Highlight | `#A35139` | Alerts, high-risk indicators |
| Deep Dark | `#1B2632` | Text, axis labels |

---

## 🔧 Tech Stack

```
Data          : pandas 2.2 · numpy 1.26 · scipy 1.13
ML            : scikit-learn 1.5 · XGBoost 2.0 · LightGBM 4.3
Forecasting   : Prophet 1.1 · statsmodels 0.14
NLP           : VADER · NLTK · transformers 4.41
Explainability: SHAP 0.45
Visualisation : matplotlib 3.9 · seaborn 0.13 · plotly 5.22
Dashboard     : Streamlit 1.35
Persistence   : joblib 1.4
```

---

## 🏛️ Indian Enterprise Context

All datasets are calibrated to the Indian BFSI sector:

- **Geography**: Pan-India distribution across 20 states, Tier-1/2/3 market segmentation
- **Currency**: ₹ values in Lakhs and Crores (standard Indian corporate convention)
- **Terminology**: CIBIL scores, RBI repo rate, GST, EBITDA, QoQ/YoY growth, FY notation
- **Institutions**: HDFC Bank, ICICI, SBI, Bajaj Finserv, TCS, Infosys referenced
- **Events**: COVID-19 economic shock (FY2020–21), UPI growth trajectory, RBI rate cycles modelled

---

## 👤 Author

**Final Year Data Science Project**  
*Domain: Finance (BFSI) | Type: Multi-Module ML Platform*  
*Stack: Python + Streamlit + sklearn + Prophet + VADER*  
*Grade Level: TCS iON / Infosys Nia Production Standard*

---

## 📄 License

This project is submitted as an academic final-year project. All synthetic data is computer-generated and does not represent real individuals or institutions.
