
import os
import streamlit as st
import pandas as pd

from app.utils.data_loader import load_csv, guess_target
from app.pages import data_exploration, visualization, prediction, about

# Page setup
st.set_page_config(page_title="Diabetes Analysis App", layout="wide")
# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧪 Data Analysis App (Course Project Scaffold)")

st.sidebar.header("Project Topic / Dataset")
topic = st.sidebar.selectbox(
    "Choose topic",
    [
        "Re-implement previous project (Train C)",
        "Dataset - Diabetes (Kaggle Pima)",
        "Dataset - Palmer Penguins (Kaggle)",
        "Dataset - THPTQG 2025 (Upload CSV)",
        "Custom dataset (Upload CSV)"
    ],
    index=1
)

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = None

if uploaded is not None:
    df = load_csv(uploaded)
    st.sidebar.success("CSV uploaded.")
else:
    st.sidebar.info("Upload CSV to start, or place a dataset file locally when deploying.")

if df is None:
    st.info("Please upload a CSV to proceed with EDA/Visualization/Modeling pages.")
    about.render()
    st.stop()

# Target selection
st.sidebar.header("Target")
t_guess = guess_target(df)
target_col = st.sidebar.selectbox("Choose target column", options=df.columns.tolist(), index=df.columns.get_loc(t_guess) if t_guess in df.columns else 0)

# Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualization", "Prediction", "About"], index=0)

if page == "Data Exploration":
    data_exploration.render(df)
elif page == "Visualization":
    visualization.render(df, target_col)
elif page == "Prediction":
    prediction.render(df, target_col)
else:
    about.render()
