
import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

from app.utils.ml_model import algorithm_options, train_and_eval
from app.utils.data_loader import apply_pima_cleaning

def render(df: pd.DataFrame, target_col: str):
    st.header("Modeling & Prediction")

    with st.expander("Preprocessing options"):
        treat_zero = st.checkbox("Treat zeros as missing (Pima)", value=True)
        if treat_zero:
            df = apply_pima_cleaning(df, True)

    st.subheader("Training")
    algo_names = list(algorithm_options().keys())
    # Prefer Stacking if available
    default_idx = algo_names.index("Stacking (RF + LR + SVC + XGB)") if "Stacking (RF + LR + SVC + XGB)" in algo_names else 0
    algo = st.selectbox("Algorithm", options=algo_names, index=default_idx)

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        scale = st.checkbox("Standardize features", value=(algo in ["LogisticRegression", "SVC (rbf)"] or algo.startswith("Stacking")))
    with col3:
        smote = st.checkbox("Use SMOTE", value=True)

    st.caption("Hyperparameter Search")
    search_type = st.selectbox("Search type", ["None", "RandomizedSearchCV", "GridSearchCV"], index=0)
    n_iter = st.number_input("RandomizedSearchCV n_iter", min_value=5, max_value=200, value=30, step=5)
    cv_folds = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1)

    if st.button("Train / Evaluate"):
        out = train_and_eval(
            df, target_col, algo_name=algo, test_size=test_size,
            scale_features=scale, use_smote=smote,
            search_type=search_type, n_iter=int(n_iter), cv_folds=int(cv_folds)
        )
        st.success("Training complete.")

        metrics = out["metrics"]
        st.subheader("Metrics")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
        cols[3].metric("F1", f"{metrics.get('f1', 0):.4f}")
        if "roc_auc" in metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        pred = out["pred"]
        X_test = out["X_test"]
        y_test = out["y_test"]

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center')
        st.pyplot(fig)

        if out["proba"] is not None and y_test.nunique()==2:
            st.subheader("ROC curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, out["proba"], ax=ax)
            st.pyplot(fig)

        # Persist to session
        st.session_state["trained_model"] = out["model"]
        st.session_state["used_features"] = out["used_features"]
        st.session_state["target_col"] = target_col

    st.subheader("Single Prediction")
    if "trained_model" not in st.session_state:
        st.warning("Train a model above first.")
        return

    used_features = st.session_state["used_features"]
    defaults = df[used_features].median(numeric_only=True).to_dict() if set(used_features).issubset(df.columns) else {c:0.0 for c in used_features}
    with st.form("predict_form"):
        inputs = {}
        cols = st.columns(3)
        for i, c in enumerate(used_features):
            with cols[i % 3]:
                val = float(defaults.get(c, 0.0))
                inputs[c] = st.number_input(c, value=val, step=0.1, format="%.3f", key=f"in_{c}")
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_one = pd.DataFrame([[inputs[c] for c in used_features]], columns=used_features)
        model = st.session_state["trained_model"]
        try:
            y_hat = model.predict(X_one)[0]
            st.success(f"Predicted class: {y_hat}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
