
import io
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional libs
_HAS_XGB = False
_HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    pass

def algorithm_options() -> Dict[str, Any]:
    algos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVC (rbf)": SVC(probability=True, kernel="rbf", random_state=42),
    }
    if _HAS_XGB:
        algos["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=0
        )
    if _HAS_LGBM:
        algos["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=-1, subsample=0.9, colsample_bytree=0.9,
            random_state=42
        )

    # Ensembles
    base = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
    ]
    if _HAS_XGB:
        base.append(("xgb", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            use_label_encoder=False, random_state=42, n_jobs=0
        )))
    algos["Voting (RF + LR + SVC + XGB)"] = VotingClassifier(estimators=base, voting="soft")
    algos["Stacking (RF + LR + SVC + XGB)"] = StackingClassifier(
        estimators=base, final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        stack_method="predict_proba", passthrough=False
    )
    return algos

def get_param_grid(name: str):
    import numpy as _np
    if name == "LogisticRegression":
        grid = {"clf__C": [0.01, 0.1, 1.0, 3.0, 10.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs", "liblinear"]}
        rand = {"clf__C": _np.logspace(-3, 2, 20).tolist(), "clf__solver": ["lbfgs", "liblinear", "saga"]}
    elif name == "RandomForest":
        grid = {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [None, 5, 10, 20], "clf__min_samples_split": [2, 5, 10]}
        rand = {"clf__n_estimators": [100, 200, 300, 400, 500, 800], "clf__max_depth": [None, 4, 6, 10, 16, 24], "clf__min_samples_split": [2, 5, 10, 20], "clf__min_samples_leaf": [1, 2, 4]}
    elif name == "SVC (rbf)":
        grid = {"clf__C": [0.1, 1.0, 3.0, 10.0], "clf__gamma": ["scale", "auto", 0.1, 0.01, 0.001]}
        rand = {"clf__C": _np.logspace(-2, 2, 10).tolist(), "clf__gamma": ["scale", "auto"] + _np.logspace(-4, -1, 6).tolist()}
    elif name == "XGBoost":
        grid = {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [3, 4, 5, 6], "clf__learning_rate": [0.03, 0.1, 0.2], "clf__subsample": [0.8, 0.9, 1.0], "clf__colsample_bytree": [0.8, 0.9, 1.0]}
        rand = {"clf__n_estimators": [100, 200, 300, 400, 600], "clf__max_depth": [2, 3, 4, 5, 6, 8], "clf__learning_rate": _np.linspace(0.02, 0.3, 10).tolist(), "clf__subsample": _np.linspace(0.7, 1.0, 7).tolist(), "clf__colsample_bytree": _np.linspace(0.7, 1.0, 7).tolist(), "clf__reg_lambda": [0.0, 0.5, 1.0, 1.5, 2.0]}
    elif name == "LightGBM":
        grid = {"clf__n_estimators": [200, 300, 500], "clf__num_leaves": [15, 31, 63], "clf__learning_rate": [0.03, 0.1, 0.2], "clf__subsample": [0.8, 0.9, 1.0], "clf__colsample_bytree": [0.8, 0.9, 1.0]}
        rand = {"clf__n_estimators": [100, 200, 300, 400, 600], "clf__num_leaves": [15, 31, 63, 127], "clf__learning_rate": _np.linspace(0.02, 0.3, 10).tolist(), "clf__subsample": _np.linspace(0.7, 1.0, 7).tolist(), "clf__colsample_bytree": _np.linspace(0.7, 1.0, 7).tolist(), "clf__reg_lambda": [0.0, 0.5, 1.0, 1.5, 2.0]}
    else:
        grid, rand = {}, {}
    return grid, rand

def build_pipeline(model_name: str, scale: bool, use_smote: bool):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale and (model_name in ["LogisticRegression", "SVC (rbf)"] or model_name.startswith("Stacking")):
        steps.append(("scaler", StandardScaler()))
    clf = algorithm_options()[model_name]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)

def train_and_eval(
    df: pd.DataFrame,
    target_col: str,
    algo_name: str,
    test_size: float = 0.2,
    scale_features: bool = True,
    use_smote: bool = True,
    search_type: str = "None",
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    used_features = X.columns.tolist()
    valid = y.notna()
    X, y = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()<=10 else None
    )
    pipe = build_pipeline(algo_name, scale_features, use_smote)

    grid_params, rand_params = get_param_grid(algo_name)
    model = pipe
    scoring = "f1" if y.nunique()==2 else "f1_macro"
    if search_type == "GridSearchCV" and grid_params:
        model = GridSearchCV(pipe, grid_params, cv=cv_folds, n_jobs=-1, scoring=scoring)
    elif search_type == "RandomizedSearchCV" and rand_params:
        model = RandomizedSearchCV(pipe, rand_params, n_iter=int(n_iter), cv=cv_folds, n_jobs=-1, random_state=random_state, scoring=scoring)

    model.fit(X_train, y_train)
    best_params = getattr(model, "best_params_", None)

    pred = model.predict(X_test)
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "best_estimator_") and hasattr(model.best_estimator_, "predict_proba"):
            proba = model.best_estimator_.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
        "recall": recall_score(y_test, pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
        "f1": f1_score(y_test, pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    }
    if proba is not None and y.nunique()==2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, proba)
        except Exception:
            pass

    return {
        "model": model,
        "metrics": metrics,
        "used_features": used_features,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "proba": proba,
        "best_params": best_params
    }
