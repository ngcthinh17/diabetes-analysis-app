
import numpy as np
import pandas as pd
from app.utils.ml_model import train_and_eval

def test_train_and_eval_logreg():
    # small synthetic dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:,0] + 0.5*X[:,1] - 0.25*X[:,2] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1","f2","f3"])
    df["Outcome"] = y
    out = train_and_eval(df, "Outcome", "LogisticRegression", test_size=0.3, scale_features=True, use_smote=False)
    assert "model" in out and "metrics" in out
    assert out["metrics"]["accuracy"] >= 0.5
