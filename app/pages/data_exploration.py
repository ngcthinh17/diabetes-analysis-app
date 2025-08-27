
import streamlit as st
import pandas as pd
import numpy as np

from app.utils.data_loader import zero_as_missing

def render(df: pd.DataFrame):
    st.header("Data Exploration")
    st.write("Preview")
    st.dataframe(df.head(50))

    st.subheader("Dtypes")
    st.write(df.dtypes)

    st.subheader("Describe (numeric)")
    st.dataframe(df.describe(include=[np.number]).T)

    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("missing_count"))

    zcols = zero_as_missing(df)
    if zcols:
        st.subheader("Zero counts (Pima-specific columns)")
        st.dataframe(df[zcols].eq(0).sum().rename("zero_count"))
    st.info("Tip: Consider treating zeros as missing for columns like Glucose, BloodPressure, SkinThickness, Insulin, BMI in Pima dataset.")
