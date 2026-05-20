# 🚀 Customer Churn Prediction System

> Production-grade Machine Learning project using XGBoost, Scikit-learn, FastAPI, and Power BI to predict customer churn probabilities and identify high-risk customers.






---

## 📌 Project Overview

Customer churn is a major business challenge in subscription-based industries. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds an end-to-end machine learning system that predicts whether a customer is likely to leave a service based on demographic information, service usage, billing details, and contract characteristics.

The system provides:

* 🔍 Churn probability prediction
* ⚠️ Risk classification (High Risk / Low Risk)
* 🌐 Real-time REST API with FastAPI
* 📊 Visual reports and business insights
* 📈 Power BI dashboard
* 🏗️ Modular MLOps-style project structure

---

## 🎯 Business Objective

The objective is to identify customers with a high likelihood of churn so organizations can proactively take retention actions such as:

* Personalized offers
* Discount campaigns
* Contract upgrades
* Customer support interventions

---

## 📂 Dataset

**IBM Telco Customer Churn Dataset**

* 7,043 customer records
* 21 predictive features
* Binary target variable: `Churn`

### Feature Categories

* Customer demographics
* Service subscriptions
* Contract information
* Payment methods
* Monthly and total charges
* Customer tenure

---

## 🛠️ Tech Stack

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

* YAML configuration
* Modular source code
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
├── tests/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

1. Data ingestion and validation
2. Data cleaning and preprocessing
3. Missing value imputation
4. One-hot encoding of categorical features
5. Feature scaling
6. Stratified train-test split
7. Baseline Random Forest model
8. XGBoost hyperparameter tuning with GridSearchCV
9. Model evaluation using ROC-AUC and Recall
10. Model serialization with Joblib
11. Real-time deployment with FastAPI

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

---

## 📊 Sample API Response

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

The dashboard includes:

* Overall churn rate
* Churn by contract type
* Churn by tenure
* Monthly charges vs churn
* High-risk customer segments
* Business recommendations

---

## 💡 Key Business Insights

* Customers on month-to-month contracts are more likely to churn.
* New customers with low tenure show significantly higher churn rates.
* Electronic check users tend to have higher churn.
* Higher monthly charges correlate with increased churn risk.

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📌 Future Improvements

* Docker containerization
* MLflow experiment tracking
* DVC for data versioning
* GitHub Actions CI/CD
* Cloud deployment (AWS, GCP, Azure)
* Model monitoring

---

## 👨‍💻 Author

**Krish Gupta**
Machine Learning | Data Science | MLOps Enthusiast

* GitHub: https://github.com/Krish-labs

---

## 📄 License

This project is licensed under the MIT License.

---

## If You Found This Project Useful

Please consider giving this repository a star on GitHub.
