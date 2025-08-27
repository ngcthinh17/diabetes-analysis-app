
import pandas as pd
from app.utils.data_loader import zero_as_missing, apply_pima_cleaning

def test_zero_as_missing_detects_columns():
    df = pd.DataFrame({
        "Glucose":[1,2,0], "BloodPressure":[0,70,80], "SkinThickness":[0,1,2], "Insulin":[0,10,20], "BMI":[0,22,33], "Outcome":[0,1,0]
    })
    cols = zero_as_missing(df)
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        assert c in cols

def test_apply_pima_cleaning_replaces_zero_with_nan():
    df = pd.DataFrame({"Glucose":[0,100], "Outcome":[0,1]})
    dff = apply_pima_cleaning(df, True)
    assert dff["Glucose"].isna().sum() == 1
