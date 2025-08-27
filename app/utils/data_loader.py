
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

PIMA_ZERO_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)

def zero_as_missing(df: pd.DataFrame) -> list:
    return [c for c in PIMA_ZERO_MISSING_COLS if c in df.columns]

def apply_pima_cleaning(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    dff = df.copy()
    if treat_zero_as_missing:
        cols = zero_as_missing(dff)
        for c in cols:
            dff[c] = dff[c].replace(0, np.nan)
    return dff

def guess_target(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Outcome", "Class", "target", "diabetes", "label"]:
        if cand in df.columns:
            return cand
    for col in df.columns:
        uniques = df[col].dropna().unique()
        if len(uniques) <= 3 and set(uniques).issubset({0, 1}):
            return col
    return df.columns[-1] if len(df.columns) else None
