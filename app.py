import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)

from src.data_loader import load_dataset, load_demo_dataset
from src.dataset_audit import (
    dataset_summary, missing_value_report, duplicate_report,
    detect_outliers_iqr, class_imbalance, validate_target_column
)
from src.quality_score import compute_quality_score
from src.model_training import train_model
from src.bias_analysis import check_bias_distribution, bias_summary
from src.fairness_metrics import group_accuracy, statistical_parity
from src.report_generator import generate_pdf_report


# ========================= Page Config =========================
st.set_page_config(page_title="Responsible AI Audit Tool", layout="wide")

st.title("🔍 Responsible AI & Bias Audit Tool")
st.write(
    """
This tool audits any dataset for:
- **Data Quality Issues**
- **Bias & Representation Skew**
- **Fairness Metrics (if classification)**
- **Model Performance**
- **Final Audit Verdict + PDF Report**
"""
)

# ========================= Buttons Section =========================
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("📂 Upload Dataset File", type=["csv", "txt", "data"])
use_demo = col2.button("📌 Use Demo Dataset (Adult Income)")

reset = st.button("🔄 Reset App")


# ========================= Reset =========================
if reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()



# ========================= Load Dataset =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# Demo dataset button
if use_demo:
    try:
        st.session_state.df = load_demo_dataset()
        st.session_state.demo_mode = True
        st.session_state.dataset_loaded = True
        st.success("✅ Demo Dataset Loaded: Adult Census Income Dataset")

    except Exception as e:
        st.error("❌ Failed to load demo dataset")
        st.exception(e)


# Uploaded dataset
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file, sep=None, engine="python")
        st.session_state.df = df_uploaded
        st.session_state.demo_mode = False
        st.session_state.dataset_loaded = True
        st.success(f"✅ Dataset Uploaded Successfully: {uploaded_file.name}")

    except Exception as e:
        st.error("❌ Failed to load uploaded dataset")
        st.exception(e)


# Restore values
df = st.session_state.df
demo_mode = st.session_state.demo_mode




# ========================= Helper: Final Decision =========================
def final_audit_decision(dataset_score, model_metrics, fairness_result):
    task = model_metrics.get("task", "unknown")

    if task == "classification":
        acc = model_metrics.get("accuracy", 0)

        if dataset_score < 60 or acc < 0.5:
            return "❌ AUDIT FAILED", "High Risk", "Not safe for deployment."
        elif dataset_score < 80 or acc < 0.7:
            return "⚠️ CONDITIONAL PASS", "Medium Risk", "Needs improvement before deployment."
        else:
            return "✅ AUDIT PASSED", "Low Risk", "Safe for controlled deployment."

    elif task == "regression":
        rmse = model_metrics.get("rmse", None)

        if rmse is None:
            return "⚠️ AUDIT INCOMPLETE", "Unknown", "Regression metrics not available."

        if dataset_score < 60 or rmse > 50:
            return "❌ AUDIT FAILED", "High Risk", "Errors too high for deployment."
        elif dataset_score < 80 or rmse > 20:
            return "⚠️ CONDITIONAL PASS", "Medium Risk", "Regression performance needs tuning."
        else:
            return "✅ AUDIT PASSED", "Low Risk", "Regression model seems usable with monitoring."

    return "⚠️ AUDIT INCOMPLETE", "Unknown Risk", "Task type could not be determined."


# ========================= Main Logic =========================
if df is not None:

    # ========================= Clean Demo Dataset =========================
    if demo_mode and "income" in df.columns:
        df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)

    # Preview (limited)
    st.subheader("📌 Dataset Preview (First 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    if demo_mode:
        st.info(
            "📌 Demo dataset is **Adult Income Census Dataset**.\n\n"
            "Suggested Target: **income**\n"
            "Suggested Sensitive Attribute: **sex** or **race**"
        )

    # ========================= Dataset Summary =========================
    st.subheader("📊 Dataset Summary")
    summary = dataset_summary(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing Columns", df.isnull().any().sum())

    with st.expander("📌 Show Column Data Types"):
        st.json(summary["dtypes"])

    # ========================= Missing Values =========================
    st.subheader("⚠ Missing Values")
    missing_report = missing_value_report(df)

    if missing_report.empty:
        st.success("✅ No missing values found.")
    else:
        st.warning("⚠ Missing values detected (Top 10 shown)")
        st.dataframe(missing_report.head(10), use_container_width=True)

    # ========================= Duplicate Rows =========================
    st.subheader("📌 Duplicate Rows Check")
    duplicates = duplicate_report(df)

    if duplicates == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠ Duplicate rows found: {duplicates}")

    # ========================= Outliers =========================
    st.subheader("🚨 Outlier / Anomaly Summary")
    outliers = detect_outliers_iqr(df)

    if len(outliers) == 0:
        st.success("✅ No major outliers detected.")
    else:
        st.warning("⚠ Outliers detected in numeric columns:")
        st.json(outliers)

    # ========================= Target and Sensitive =========================
    st.subheader("🎯 Select Target & Sensitive Attribute")

    # Default selections for demo dataset
    default_target_index = 0
    default_sensitive_index = 0

    if demo_mode:
        if "income" in df.columns:
            default_target_index = df.columns.get_loc("income")
        if "sex" in df.columns:
            default_sensitive_index = df.columns.get_loc("sex")

    target_col = st.selectbox(
        "Select Target Column (Prediction Label)",
        df.columns,
        index=default_target_index
    )

    sensitive_col = st.selectbox(
        "Select Sensitive Attribute Column (Bias Check)",
        df.columns,
        index=default_sensitive_index
    )

    # Validate Target
    valid, msg = validate_target_column(df, target_col)
    if not valid:
        st.error(f"❌ Invalid Target Column: {msg}")
        st.stop()
    else:
        st.success(f"✅ Target Column Valid: {msg}")

    # ========================= Class Imbalance =========================
    st.subheader("⚖ Class Imbalance Check")
    imbalance = class_imbalance(df, target_col)

    if isinstance(imbalance, dict) and "error" in imbalance:
        st.info("Class imbalance not applicable (Regression dataset).")
    else:
        st.write("Target Distribution (Top Classes):")
        st.json(imbalance)

    # ========================= Quality Score =========================
    st.subheader("✅ Dataset Quality Score")
    quality_result = compute_quality_score(df, missing_report, duplicates, outliers, imbalance)

    col1, col2 = st.columns(2)
    col1.metric("Dataset Quality Score", f"{quality_result['quality_score']} / 100")
    col2.metric("Risk Level", quality_result["risk_level"])

    st.write("📌 Suitability:", quality_result["suitability"])

    if quality_result["recommendations"]:
        with st.expander("📌 Recommended Fixes"):
            for rec in quality_result["recommendations"]:
                st.write("🔹", rec)

    # ========================= Train Model =========================
    st.subheader("🤖 Train Model & Evaluate Performance")

    model, X_test, y_test, problem_type = train_model(df, target_col)
    y_pred = model.predict(X_test)

    # ========================= Metrics =========================
    st.subheader("📈 Model Performance Summary")

    if problem_type == "classification":
        metrics = {
            "task": "classification",
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)
        }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", metrics["accuracy"])
        c2.metric("Precision", metrics["precision"])
        c3.metric("Recall", metrics["recall"])
        c4.metric("F1 Score", metrics["f1_score"])

    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "task": "regression",
            "mae": round(mae, 3),
            "rmse": round(rmse, 3)
        }

        c1, c2 = st.columns(2)
        c1.metric("MAE", metrics["mae"])
        c2.metric("RMSE", metrics["rmse"])

    # ========================= Bias Distribution =========================
    st.subheader("⚖ Bias Distribution Audit (Summary)")

    bias_table = None
    bias_info = None

    if problem_type == "classification":
        bias_table = check_bias_distribution(df, sensitive_col, target_col)

        if bias_table is None or bias_table.empty:
            st.warning("Bias distribution could not be generated.")
        else:
            bias_info = bias_summary(bias_table)

            st.write(f"📌 Worst Affected Group: **{bias_info['worst_group']}**")
            st.write(f"📌 Max Disparity: **{bias_info['max_disparity']}%**")
            st.write(f"📌 Bias Risk Level: **{bias_info['risk']}**")

            with st.expander("Show Bias Table (Optional)"):
                st.dataframe(bias_table.head(10), use_container_width=True)

    else:
        st.info("Bias distribution not applicable for regression datasets.")

    # ========================= Fairness Metrics =========================
    st.subheader("📊 Fairness Audit Summary")

    fairness_results = {"note": "Fairness audit not applicable"}

    if problem_type == "classification" and df[target_col].nunique() == 2:

        df_test = X_test.copy()
        df_test[target_col] = y_test
        df_test["y_pred"] = y_pred

        if sensitive_col not in df_test.columns:
            st.warning(
                f"⚠ Sensitive attribute '{sensitive_col}' is not in X_test. "
                "Fairness evaluation cannot run."
            )
        else:
            group_acc = group_accuracy(df_test, sensitive_col, target_col, "y_pred")
            parity = statistical_parity(df_test, sensitive_col, "y_pred")

            fairness_results = {
                "Group Accuracy": group_acc,
                "Statistical Parity (Selection Rate)": parity
            }

            if isinstance(parity, dict) and "error" not in parity:
                st.write(f"📌 Selection Rate Difference: **{parity.get('difference', 'N/A')}**")

                if parity.get("difference") is not None and parity["difference"] > 0.2:
                    st.error("❌ Fairness Risk Detected (High disparity across groups).")
                else:
                    st.success("✅ Fairness looks acceptable (low disparity).")

                with st.expander("Show Fairness Details (Optional)"):
                    st.json(fairness_results)

            else:
                st.warning("Fairness audit could not be completed.")
                st.json(parity)

    else:
        st.info("Fairness audit requires binary classification (2 target classes). Not applicable here.")

    # ========================= Final Audit Decision =========================
    st.subheader("🏁 Final Audit Verdict")

    verdict, risk, note = final_audit_decision(
        quality_result["quality_score"],
        metrics,
        fairness_results
    )

    col1, col2 = st.columns(2)
    col1.metric("Final Verdict", verdict)
    col2.metric("Risk Level", risk)

    st.write("📌 Recommendation:", note)

    # ========================= Generate Report =========================
    st.subheader("📄 Generate Audit Report (PDF)")

    # initialize session state for pdf
    if "pdf_bytes" not in st.session_state:
        st.session_state["pdf_bytes"] = None

    if "report_ready" not in st.session_state:
        st.session_state["report_ready"] = False


    generate_btn = st.button("Generate Report")

    if generate_btn:
        try:
            with st.spinner("⏳ Generating PDF Report... Please wait"):
                pdf_bytes = generate_pdf_report(
                    "reports/audit_report.pdf",
                    quality_result["quality_score"],
                    quality_result,
                    metrics,
                    fairness_results,
                    bias_table
                )

            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["report_ready"] = True

            st.success("✅ Report Generated Successfully!")

        except Exception as e:
            st.session_state["pdf_bytes"] = None
            st.session_state["report_ready"] = False

            st.error("❌ Report generation failed!")
            st.exception(e)


    # show download button after report is generated
    if st.session_state["report_ready"] and st.session_state["pdf_bytes"] is not None:
        st.download_button(
            label="⬇️ Download Report",
            data=st.session_state["pdf_bytes"],
            file_name="audit_report.pdf",
            mime="application/pdf"
        )
