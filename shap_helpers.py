# shap_helpers.py
import shap
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


def plot_summary(explainer, X_test):
    #fig = plt.figure(figsize=(10,8))
    shap_values = explainer.shap_values(X_test)
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig = plt.gcf()
    #ax.scatter([1, 2, 3], [1, 2, 3])
    # other plotting actions...
    #st.pyplot(fig)
    st.pyplot(fig,bbox_inches='tight')
    plt.clf()

def plot_dependence(explainer, X_test, feature):
    #fig = plt.figure(figsize=(15,15))
    shap_values = explainer.shap_values(X_test)
    st.subheader(f"Dependence Plot for {feature}")
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    fig = plt.gcf()
    #st.pyplot(fig)
    st.pyplot(fig,bbox_inches='tight')
    plt.clf()

def plot_waterfall(explainer, X_test, index):
    #fig = plt.figure(figsize=(10,8))
    shap_values = explainer.shap_values(X_test)
    st.subheader("Waterfall Plot for selected transaction")
    plt.clf()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[index], X_test.iloc[index]
    )
    
    fig = plt.gcf()
    # st.pyplot(fig)
    st.pyplot(fig,bbox_inches='tight')
    plt.clf()

def plot_force(explainer, X_test, index):
    fig = plt.figure(figsize=(10,8))
    shap_values = explainer.shap_values(X_test)
    st.subheader("Force Plot (opens in browser tab)")
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X_test.iloc[index],
        matplotlib=False
    )
    shap.save_html("force_plot.html", force_html)
    st.markdown("[Open Force Plot](force_plot.html)")
