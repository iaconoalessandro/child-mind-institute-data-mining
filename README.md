# Child Mind Institute: Problematic Internet Use Prediction

This repository contains the refactored, modular data mining project for predicting **Problematic Internet Use (PIU)** in children and adolescents based on physical health measurements, sleep assessment questionnaires, and wrist-worn accelerometer time series.

---

## 📁 Repository Overview

The repository is structured into two main sub-projects:

### 1. 📊 [Tabular Data Analysis (`tabular/`)](./tabular)
Focuses on predicting the **Severity Impairment Index (`sii`)** using non-PCIAT tabular features including body measurements, physical fitness tests, bio-impedance readings, and sleep disturbance questionnaires.
- **Key Modules**: Exploratory Data Analysis (EDA), Missing Value Imputation, Outlier Detection, Regression & Classification Pipelines, Model Tuning (Optuna), SHAP & Feature Importance.
- **Details & Documentation**: See [`tabular/README.md`](./tabular/README.md).

### 2. ⌚ [Time Series Analysis (`timeseries/`)](./timeseries)
Focuses on predicting problematic internet use directly from **wrist sensor accelerometer recordings** (4,437 subjects, ~4.2 days of activity aggregated into 30-minute intervals).
- **Key Modules**: Preprocessing & Outlier Detection, Time Series Representations & PULSAR Feature Extraction, Clustering & Matrix Profile, Shapelet Mining, Sequential Pattern Mining (SPM), Classification Models & SHAP Explainability.
- **Details & Documentation**: See [`timeseries/README.md`](./timeseries/README.md).

---

## 📑 Reports & Guidelines

- 📄 [`Project_DM2_Iacono_Mungo.pdf`](./Project_DM2_Iacono_Mungo.pdf): Final Data Mining project report by Alessandro Iacono & Mungo.
- 📋 [`Project Guidelines.pdf`](./Project Guidelines.pdf): Course project specifications and requirements.

---

## 🚀 Getting Started

To run either sub-project, navigate into its respective folder and install the dependencies:

```bash
# Tabular project
cd tabular
pip install -r requirements.txt
python main.py

# Time series project
cd ../timeseries
pip install -r requirements.txt
python main.py
```
