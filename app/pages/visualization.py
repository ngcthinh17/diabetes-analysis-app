
import streamlit as st
import pandas as pd
import numpy as np

from app.utils.visualization import plot_histogram, plot_boxplot, plot_corr

def render(df: pd.DataFrame, target_col: str):
    st.header("Visualization")
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in cols_num:
        cols_num.remove(target_col)

    if not cols_num:
        st.warning("No numeric columns to visualize.")
        return

    left, right = st.columns(2)
    with left:
        col_hist = st.selectbox("Histogram column", options=cols_num, index=0)
        st.pyplot(plot_histogram(df, col_hist))
    with right:
        col_box = st.selectbox("Boxplot column", options=cols_num, index=min(1, len(cols_num)-1))
        st.pyplot(plot_boxplot(df, col_box))

    st.subheader("Correlation heatmap")
    chosen_corr = st.multiselect("Choose numeric columns", options=cols_num, default=cols_num[:min(8, len(cols_num))])
    if len(chosen_corr) >= 2:
        st.pyplot(plot_corr(df, chosen_corr))
    else:
        st.info("Pick at least 2 columns to draw heatmap.")
