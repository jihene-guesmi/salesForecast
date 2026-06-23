# Predictive Time-Series Forecasting Framework and Computational Inference Pipeline

An end-to-end machine learning system designed for **deterministic temporal sales forecasting**, engineered with an integrated pipeline utilizing **FastAPI, Pydata Stack, and Scikit-learn**.

## 🔬 Research & Project Overview

Predictive modeling for temporal business matrices requires robust feature engineering and regression optimization to capture underlying trends, seasonal fluctuations, and cyclical variations. This project introduces a production-grade machine learning pipeline that ingests historical tabular data, processes time-series dependencies, and optimizes regression models for accurate horizon forecasting.

By pairing a feature-engineered machine learning backbone with a decoupled ASGI inference framework, this repository demonstrates a highly scalable blueprint for deploying structural forecasting models into production environments.

### Key Technical Contributions:
- **Temporal Feature Engineering:** Extraction of rolling statistics, seasonal lags, and cyclical time components to capture complex temporal dynamics.
- **Asynchronous API Deployment:** Packaging the finalized serialized regression models into a high-throughput FastAPI architecture for sub-millisecond execution.
- **Static Dashboard Integration:** A lightweight front-end visualizer rendering asynchronous predictions alongside historical baselines for comparative trend evaluation.

---

## 🏗️ System Architecture & Data Flow

Use code with caution.[Historical Sales Matrix] ──> [Temporal Feature Engineering Layer]│▼[Optimized Regression Ensemble]│▼[Serialized Model Artifact (.pkl)]│▼[FastAPI ASGI Microservice]│▼[HTML5/CSS3 Analytical Interface]
---

## 🛠️ Computational Tech Stack

- **Data Engineering Suite:** NumPy, Pandas, Scikit-learn
- **Inference Service Pipeline:** FastAPI (Uvicorn ASGI Gateway)
- **Visualization & Environment:** Jupyter Notebooks, HTML5, CSS3, JavaScript

---

## 📂 Repository Structure

```text
sales_app/
│
├── app.py              # FastAPI Application Server & Computational Routing Layers
├── model.pkl           # Serialized Predictive Model Weights & Scaling Matrices
├── index.html          # HTML5 Frontend Interface Panel & Data Visualizer
├── README.md           # Core Documentation
│
└── notebooks/          # Experimental Prototyping Directory
    ├── EDA_Analytics.ipynb         # Exploratory Data Analysis & Feature Distributions
    └── Model_Training_Loops.ipynb  # Hyperparameter Optimization & Evaluation Matrices
```

---

## 🚀 Core Research Features & Capabilities

- **Cyclical Encoding & Normalization:** Automated transformation of raw dates into highly predictive mathematical features (lags, rolling averages, seasonality indicators).
- **Sub-Millisecond Inference Gateway:** Microservice architecture designed for ultra-low latency request-response loops.
- **Static Statistical Projections:** Interactive dashboard rendering deterministic prediction values against past trends to aid algorithmic auditing.

---

## 📊 Evaluation & Baseline Metrics

The predictive backbones were benchmarked against standard continuous evaluation matrices across a strict validation split:

| Algorithmic Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-Squared ($R^2$) |
| :--- | :--- | :--- | :--- |
 |

