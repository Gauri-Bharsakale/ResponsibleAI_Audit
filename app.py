# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# from src.quality_score import compute_quality_score

# from src.dataset_audit import (
#     dataset_summary,
#     missing_value_report,
#     duplicate_report,
#     detect_outliers_iqr,
#     class_imbalance
# )

# st.set_page_config(page_title="Dataset Auditor", layout="wide")

# st.title("📊 Responsible AI Dataset Auditor Tool")
# st.write("Upload any dataset file (CSV/TXT) to audit anomalies, missing values, imbalance, and outliers.")

# uploaded_file = st.file_uploader("Upload Dataset File", type=["csv", "txt"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except Exception:
#         df = pd.read_csv(uploaded_file, delimiter=r"\s+", engine="python")

#     st.success("✅ Dataset Uploaded Successfully!")

#     # Show dataset preview
#     st.subheader("🔍 Dataset Preview")
#     st.dataframe(df.head(20))

#     # Dataset summary
#     st.subheader("📌 Dataset Summary")
#     summary = dataset_summary(df)

#     col1, col2 = st.columns(2)
#     col1.metric("Rows", summary["rows"])
#     col2.metric("Columns", summary["columns"])

#     st.write("### Column Data Types")
#     st.json(summary["dtypes"])

#     # Missing values
#     st.subheader("⚠️ Missing Value Report")
#     missing_report = missing_value_report(df)

#     if missing_report.empty:
#         st.success("✅ No missing values found!")
#     else:
#         st.warning("⚠️ Missing values detected!")
#         st.dataframe(missing_report)

#     # Duplicate rows
#     st.subheader("📌 Duplicate Rows")
#     duplicates = duplicate_report(df)

#     if duplicates == 0:
#         st.success("✅ No duplicate rows found!")
#     else:
#         st.warning(f"⚠️ Duplicate rows found: {duplicates}")

#     # Outliers
#     st.subheader("🚨 Outlier / Anomaly Detection (IQR Method)")
#     outliers = detect_outliers_iqr(df)

#     if len(outliers) == 0:
#         st.success("✅ No major outliers detected in numeric columns!")
#     else:
#         st.warning("⚠️ Outliers detected in these columns:")
#         st.write(outliers)

#     # Target column imbalance
#     st.subheader("⚖️ Class Imbalance Check")
#     target_col = st.text_input("Enter Target Column Name (optional)", "")

#     imbalance = None  # IMPORTANT: default value

#     if target_col:
#         imbalance = class_imbalance(df, target_col)

#         if imbalance is None:
#             st.error("❌ Target column not found in dataset!")
#         else:
#             st.write("### Target Distribution (%)")
#             st.write(imbalance)

#             fig, ax = plt.subplots()
#             imbalance.plot(kind="bar", ax=ax)
#             ax.set_title("Target Class Distribution")
#             ax.set_ylabel("Percentage")
#             st.pyplot(fig)

#     # Correlation analysis
#     st.subheader("📈 Correlation Check (Numeric Columns)")
#     numeric_df = df.select_dtypes(include=["int64", "float64"])

#     high_corr_pairs = []
#     high_corr_count = 0

#     if numeric_df.shape[1] > 1:
#         corr = numeric_df.corr()

#         st.write("### Correlation Matrix")
#         st.dataframe(corr)

#         for i in range(len(corr.columns)):
#             for j in range(i + 1, len(corr.columns)):
#                 if abs(corr.iloc[i, j]) > 0.85:
#                     high_corr_pairs.append(
#                         (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3))
#                     )

#         high_corr_count = len(high_corr_pairs)

#         if high_corr_count > 0:
#             st.warning("⚠️ Highly correlated features detected (possible redundancy/leakage):")
#             st.write(high_corr_pairs)
#         else:
#             st.success("✅ No extremely high correlations detected.")
#     else:
#         st.info("Not enough numeric columns for correlation analysis.")

#     # ==============================
#     # FINAL DATASET QUALITY VERDICT
#     # ==============================
#     st.subheader("✅ Final Dataset Conclusion (Audit Verdict)")

#     audit_result = compute_quality_score(
#         df=df,
#         missing_report=missing_report,
#         duplicates=duplicates,
#         outliers=outliers,
#         imbalance=imbalance,
#         high_corr_count=high_corr_count
#     )

#     st.metric("Dataset Quality Score", f"{audit_result['quality_score']} / 100")
#     st.write("### Risk Level:", audit_result["risk_level"])
#     st.write("### Suitability:", audit_result["suitability"])

#     if len(audit_result["risk_flags"]) > 0:
#         st.warning("⚠️ Issues Detected:")
#         for flag in audit_result["risk_flags"]:
#             st.write(f"- {flag}")

#     if len(audit_result["recommendations"]) > 0:
#         st.info("🛠 Recommended Fixes:")
#         for rec in audit_result["recommendations"]:
#             st.write(f"- {rec}")
#     else:
#         st.success("🎉 Dataset looks clean and ready for ML training!")







import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)

from src.data_loader import load_dataset
from src.dataset_audit import (
    dataset_summary, missing_value_report, duplicate_report,
    detect_outliers_iqr, class_imbalance, validate_target_column
)
from src.quality_score import compute_quality_score
from src.model_training import train_model
from src.bias_analysis import check_bias_distribution
from src.fairness_metrics import group_accuracy, statistical_parity
from src.report_generator import generate_pdf_report


st.set_page_config(page_title="Responsible AI Audit Tool", layout="wide")

st.title("🔍 Responsible AI & Bias Audit Tool")
st.write("Upload any dataset to audit data quality, bias, fairness, and generate an audit report.")

uploaded_file = st.file_uploader("Upload Dataset File", type=["csv", "txt", "data"])


if uploaded_file:
    df = load_dataset(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    # Show only first 10 rows (avoid huge tables)
    st.subheader("📌 Dataset Preview (First 10 rows)")
    st.dataframe(df.head(10))

    # ========================= Dataset Summary =========================
    st.subheader("📌 Dataset Summary")
    summary = dataset_summary(df)

    col1, col2 = st.columns(2)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])

    with st.expander("Show Data Types"):
        st.json(summary["dtypes"])

    # ========================= Missing Values =========================
    st.subheader("⚠ Missing Value Report")
    missing_report = missing_value_report(df)

    if missing_report.empty:
        st.success("✅ No missing values found.")
    else:
        st.warning("⚠ Missing values detected (Top 10 shown)")
        st.dataframe(missing_report.head(10))

    # ========================= Duplicate Rows =========================
    st.subheader("📌 Duplicate Rows")
    duplicates = duplicate_report(df)
    st.write("Duplicate rows found:", duplicates)

    # ========================= Outliers =========================
    st.subheader("🚨 Outlier Summary")
    outliers = detect_outliers_iqr(df)

    if len(outliers) == 0:
        st.success("✅ No major outliers detected.")
    else:
        st.warning("⚠ Outliers detected (Summary)")
        st.json(outliers)

    # ========================= Target and Sensitive =========================
    st.subheader("🎯 Select Target & Sensitive Attribute")
    target_col = st.selectbox("Select Target Column", df.columns)
    sensitive_col = st.selectbox("Select Sensitive Attribute Column", df.columns)

    # Validate Target Column
    valid, msg = validate_target_column(df, target_col)
    if not valid:
        st.error(f"❌ Invalid Target Column: {msg}")
        st.stop()
    else:
        st.success(f"✅ Target Column Valid: {msg}")

    # ========================= Class Imbalance =========================
    st.subheader("⚖ Class Imbalance")
    imbalance = class_imbalance(df, target_col)

    if "error" in imbalance:
        st.warning("Class imbalance not applicable (Regression dataset).")
    else:
        st.json(imbalance)

    # ========================= Quality Score =========================
    st.subheader("✅ Dataset Quality Score")
    quality_result = compute_quality_score(df, missing_report, duplicates, outliers, imbalance)

    st.metric("Dataset Quality Score", f"{quality_result['quality_score']} / 100")
    st.write("Risk Level:", quality_result["risk_level"])
    st.write("Suitability:", quality_result["suitability"])

    if quality_result["recommendations"]:
        st.info("Recommendations:")
        for rec in quality_result["recommendations"]:
            st.write("-", rec)

    # ========================= Train Model =========================
    st.subheader("🤖 Train Model and Evaluate")

    model, X_test, y_test, problem_type = train_model(df, target_col)

    y_pred = model.predict(X_test)

    # ========================= Metrics =========================
    st.subheader("📊 Model Performance Metrics")

    if problem_type == "classification":
        metrics = {
            "task": "classification",
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)
        }
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "task": "regression",
            "mae": round(mae, 3),
            "rmse": round(rmse, 3)
        }

    st.json(metrics)

    # ========================= Bias Distribution =========================
    st.subheader("⚖ Bias Distribution Summary")

    bias_table = None
    if problem_type == "classification":
        bias_table = check_bias_distribution(df.dropna(), sensitive_col, target_col)

        if bias_table is None or bias_table.empty:
            st.warning("Bias distribution could not be generated.")
        else:
            st.write("Showing only first 10 rows of bias distribution:")
            st.dataframe(bias_table.head(10))
    else:
        st.info("Bias distribution not applicable for regression datasets.")

    # ========================= Fairness Metrics =========================
    st.subheader("📊 Fairness Metrics")

    fairness_results = {"note": "Fairness audit not applicable"}

    if problem_type == "classification" and df[target_col].nunique() == 2:
        df_test = X_test.copy()
        df_test[target_col] = y_test
        df_test["y_pred"] = y_pred

        group_acc = group_accuracy(df_test, sensitive_col, target_col, "y_pred")
        parity = statistical_parity(df_test, sensitive_col, "y_pred")

        fairness_results = {
            "Group Accuracy": group_acc,
            "Statistical Parity (Selection Rate)": parity
        }

        st.json(fairness_results)

    else:
        st.info("Fairness audit requires binary classification target (2 classes). Not applicable here.")

    # ========================= Generate Report =========================
    st.subheader("📄 Generate Audit Report (PDF)")

    if st.button("Generate Report"):
        report_path = generate_pdf_report(
            "reports/audit_report.pdf",
            quality_result["quality_score"],
            quality_result,
            metrics,
            fairness_results,
            bias_table
        )

        st.success("✅ Audit Report Generated Successfully!")

        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name="audit_report.pdf")
