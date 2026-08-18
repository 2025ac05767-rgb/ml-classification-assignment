import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Breast Cancer Classification App")

st.write(
    "Upload the test CSV file and select a machine learning model "
    "to view predictions and evaluation metrics."
)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
# Load models
models = {
    "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
    "Decision Tree": joblib.load(MODEL_DIR / "decision_tree.pkl"),
    "kNN": joblib.load(MODEL_DIR / "knn.pkl"),
    "Naive Bayes": joblib.load(MODEL_DIR / "naive_bayes.pkl"),
    "Random Forest": joblib.load(MODEL_DIR / "random_forest.pkl")
}

scaler = joblib.load(MODEL_DIR / "scaler.pkl")

uploaded_file = st.file_uploader(
    "Upload test data CSV",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("CSV must contain a 'target' column.")
    else:
        X_test = df.drop(columns=["target"])
        y_test = df["target"]

        model = models[model_name]

        # Apply scaling only to models that were trained on scaled data
        if model_name in ["Logistic Regression", "kNN"]:
            X_input = scaler.transform(X_test)
        else:
            X_input = X_test

        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ],
            "Score": [
                accuracy,
                auc,
                precision,
                recall,
                f1,
                mcc
            ]
        })

        st.dataframe(metrics_df)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(cm_df)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())