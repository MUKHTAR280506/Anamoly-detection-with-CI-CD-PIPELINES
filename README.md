# 💳 Fraud Anomaly Detection Dashboard

An interactive, explainable AI-powered dashboard built with **Streamlit**, **XGBoost**, and **SHAP** 
to detect and interpret fraudulent transactions using synthetic data. 
This project demonstrates the full lifecycle of a data science app, from data generation and
model training to deployment and CI/CD automation.

URL to access the model-
https://anamoly-detection-with-ci-cd-pipelines-etvk3us6aaezk8sxqe2qii.streamlit.app


---

## 📁 Project Structure

```
anamolydetection/
├── app.py                         # Main Streamlit dashboard app
├── synthetic_dataset.py          # Script to generate synthetic fraud data
├── shap_helpers.py               # SHAP visualization helper functions
├── models/
│   ├── xgb_fraud_model.pkl       # Trained XGBoost model
│   └── shap_explainer.pkl        # SHAP explainer object
├── datasets/
│   ├── synthetic_fraud_data.csv  # Full dataset used for training/testing
│   └── X_test.csv                # Test data used for SHAP explanations
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit config (headless, port, CORS)
└── .github/
    └── workflows/
        └── deploy.yml            # CI/CD GitHub Actions workflow
```

---

## 🚀 Features

* ✅ **Synthetic fraud data generation**
* ✅ **XGBoost model training**
* ✅ **Model explainability with SHAP**
* ✅ **Interactive Streamlit dashboard**
* ✅ **Dependence and waterfall plots**
* ✅ **GitHub Actions CI/CD pipeline**
* ✅ **Ready for Streamlit Cloud deployment**

---

## 📊 Key Model Features

```python
model_features = [
    "transaction_amount", "transaction_type", "device_type", "location_distance_km",
    "is_foreign_transaction", "is_high_risk_country", "is_weekend", "hour_of_day",
    "previous_fraud_flag", "avg_transaction_amount_user", "is_large_transaction",
    "time_since_last_txn", "is_night", "location_change_flag", "device_usage_count"
]
```

---

## 🧪 Setup & Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/MUKHTAR280506/Anamoly-detection-with-CI-CD-PIPELINES.git
cd Anamoly-detection-with-CI-CD-PIPELINES
```

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push your project to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo.
4. Set the main file as `app.py` and define `requirements.txt`.
5. Optionally define secrets in the Streamlit Cloud UI.

---

## 🔄 CI/CD with GitHub Actions

This project includes a CI workflow in `.github/workflows/deploy.yml` which:

* Installs dependencies
* Validates the Streamlit app
* Ensures the dashboard can start on a headless server

Runs automatically on push to the `main` branch.

---

## 🧠 Explainability with SHAP

The app supports:

* **SHAP Summary Plot**
* **Dependence Plot**
* **Waterfall Plot (per instance)**

Helps users and auditors understand **why** a transaction was flagged.

## 📌 Future Enhancements

* ✅ Real-time fraud detection with streaming data
* ✅ Role-based access control
* ✅ PostgreSQL or MongoDB integration
* ✅ Alerting or notifications for anomalies

---

## 🙌 Acknowledgements

* [SHAP](https://github.com/slundberg/shap)
* [XGBoost](https://xgboost.readthedocs.io/)
* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)

---

## 📧 Contact

**Mukhtar Ahmad**
[LinkedIn](https://www.linkedin.com/in/mukhtar280506)
Email: [mukhtar.wimc@gmail.com](mailto:mukhtar.wimc@gmail.com)

