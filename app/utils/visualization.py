
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Histogram - {col}")
    ax.set_xlabel(col); ax.set_ylabel("Frequency")
    return fig

def plot_boxplot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna(), vert=True, labels=[col])
    ax.set_title(f"Boxplot - {col}")
    return fig

def plot_corr(df: pd.DataFrame, cols: List[str]):
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig
