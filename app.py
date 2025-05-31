# app.py
import streamlit as st
import pandas as pd
import joblib
from shap_helpers import plot_summary, plot_dependence, plot_waterfall, plot_force

# Load model and explainer
model = joblib.load(r"models/xgb_fraud_model.pkl")
explainer = joblib.load(r"models/shap_explainer.pkl")

# Load sample test data
X_test = pd.read_csv(r"datasets/X_test.csv")  # Save X_test earlier

st.set_page_config(layout="wide")
st.title("🔍 Fraud Detection & SHAP Explainability Dashboard version 1.0")

# Sidebar: Choose index
idx = st.sidebar.slider("Select Transaction Index", 0, len(X_test)-1, 0)
input_data = X_test.iloc[[idx]]

# Predict
pred_proba = model.predict_proba(input_data)[0][1]
pred = model.predict(input_data)[0]

st.subheader("💡 Prediction")
st.write(f"**Fraud Probability:** {pred_proba:.4f}")
st.write(f"**Predicted Label:** {'Fraud' if pred == 1 else 'Not Fraud'}")

# Tabs for SHAP visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Dependence", "Waterfall", "Force"])

with tab1:
    plot_summary(explainer, X_test)

with tab2:
    feature = st.selectbox("Select feature", X_test.columns)
    plot_dependence(explainer, X_test, feature)

with tab3:
    plot_waterfall(explainer, X_test, idx)

with tab4:
    plot_force(explainer, X_test, idx)
