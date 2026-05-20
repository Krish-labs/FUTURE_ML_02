# 🚀 Customer Churn Prediction System

> Production-grade Machine Learning system that predicts customer churn using XGBoost, Scikit-learn, FastAPI, and Power BI.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Overview

Customer churn is a critical business problem in subscription-based industries. This project implements an end-to-end machine learning solution that predicts whether a customer is likely to leave a service based on demographic information, service usage, billing details, and contract characteristics.

### Key Features

* 🔍 Churn probability prediction
* ⚠️ Risk classification (High Risk / Low Risk)
* 🌐 Real-time REST API using FastAPI
* 📊 Visual reports and business insights
* 📈 Power BI dashboard
* 🏗️ Modular MLOps-style project architecture

---

## 🎯 Business Objective

The objective is to identify high-risk customers early so businesses can take proactive retention actions such as personalized offers, contract upgrades, and targeted support interventions.

---

## 📂 Dataset

**IBM Telco Customer Churn Dataset**

* 7,043 customer records
* 21 predictive features
* Binary target variable: `Churn`

### Feature Categories

* Demographic information
* Service subscriptions
* Contract details
* Billing and payment methods
* Customer tenure

---

## 🛠️ Technology Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

### Deployment

* FastAPI
* Uvicorn

### Visualization

* Power BI

### MLOps Practices

* YAML-based configuration
* Modular source code organization
* Joblib model serialization
* Automated preprocessing pipelines

---

## 📁 Project Structure

```text
FUTURE_ML_02/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── Customer_ChurnDataset.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── churn_model.joblib
│   └── preprocessor.joblib
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       └── correlation_matrix.png
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── notebooks/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

1. Data ingestion and validation
2. Data preprocessing and cleaning
3. Missing value imputation
4. Categorical encoding
5. Feature scaling
6. Stratified train-test split
7. Baseline model development
8. Hyperparameter tuning using GridSearchCV
9. Model evaluation using ROC-AUC and Recall
10. Model serialization with Joblib
11. Real-time API deployment with FastAPI

---

## 🤖 Model Details

### Baseline Model

* Random Forest Classifier

### Final Model

* XGBoost Classifier

### Evaluation Metrics

* ROC-AUC Score
* Recall
* Precision
* F1-Score
* Confusion Matrix
* Classification Report

---

## 📊 Sample Prediction Response

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 85.31,
  "risk_level": "High Risk"
}
```

---

## 🚀 Installation

```bash
git clone https://github.com/Krish-labs/FUTURE_ML_02.git
cd FUTURE_ML_02

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

```bash
python src/train.py
```

### Start the API

```bash
python app.py
```

### Open Interactive API Documentation

```text
http://127.0.0.1:8000/docs
```

---

## 📊 Power BI Dashboard

The dashboard provides:

* Overall churn rate
* Churn by contract type
* Churn by tenure
* Monthly charges vs. churn
* High-risk customer segments
* Actionable business recommendations

---

## 💡 Key Business Insights

* Customers on month-to-month contracts have the highest churn risk.
* New customers with low tenure are more likely to churn.
* Electronic check users show elevated churn rates.
* Higher monthly charges correlate with increased churn probability.

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📌 Future Enhancements

* Docker containerization
* MLflow experiment tracking
* DVC for data and model versioning
* GitHub Actions CI/CD
* Cloud deployment (AWS, GCP, Azure)
* Model monitoring and alerting

---

## 👨‍💻 Author

**Krish Gupta**
Machine Learning | Data Science | MLOps Enthusiast

* GitHub: https://github.com/Krish-labs

---

## 📄 License

This project is licensed under the MIT License.
