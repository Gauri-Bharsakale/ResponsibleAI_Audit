# from fpdf import FPDF

# def generate_report(metrics, output_path):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     pdf.cell(0, 10, "Responsible AI Audit Report", ln=True)

#     for k, v in metrics.items():
#         pdf.cell(0, 10, f"{k}: {v:.3f}", ln=True)

#     pdf.output(output_path)



from fpdf import FPDF
import os


def clean_text(text):
    """
    Removes unsupported unicode characters for FPDF (latin-1).
    """
    if text is None:
        return ""
    return str(text).encode("latin-1", "ignore").decode("latin-1")


def verdict_from_score(score):
    if score >= 80:
        return "PASS", "Low Risk"
    elif score >= 60:
        return "CONDITIONAL PASS", "Medium Risk"
    else:
        return "FAIL", "High Risk"


def model_verdict(metrics):
    """
    Handles both classification and regression metrics.
    """
    task = metrics.get("task", "unknown")

    if task == "classification":
        acc = metrics.get("accuracy", None)

        if acc is None:
            return (
                "Model evaluation could not be performed.",
                "Accuracy score was not available.",
                "Check target column selection and dataset format."
            )

        if acc < 0.5:
            return (
                "Model performance is poor.",
                "The model is not learning meaningful patterns.",
                "Do NOT deploy. Improve data quality and features."
            )
        elif acc < 0.7:
            return (
                "Model performance is moderate.",
                "Model predictions may be unstable.",
                "Improve feature engineering and tune hyperparameters."
            )
        else:
            return (
                "Model performance is strong.",
                "Predictions are reasonably reliable.",
                "Model is suitable for controlled deployment."
            )

    elif task == "regression":
        mae = metrics.get("mae", None)
        rmse = metrics.get("rmse", None)

        if mae is None or rmse is None:
            return (
                "Regression evaluation incomplete.",
                "MAE/RMSE scores were missing.",
                "Ensure target column is numeric and dataset is clean."
            )

        if rmse > 50:
            return (
                "Regression model performance is weak.",
                "Prediction errors are very high.",
                "Not deployable. Improve dataset and preprocessing."
            )
        elif rmse > 20:
            return (
                "Regression model performance is moderate.",
                "Errors are noticeable.",
                "Tuning and feature engineering recommended."
            )
        else:
            return (
                "Regression model performance is strong.",
                "Errors are within acceptable range.",
                "Model can be deployed with monitoring."
            )

    else:
        return (
            "Model evaluation not supported.",
            "Unknown ML task type.",
            "Ensure correct target selection."
        )


def fairness_verdict(fairness_result):
    """
    Fairness result may contain errors or parity outputs.
    """

    if fairness_result is None:
        return (
            "Fairness could not be evaluated.",
            "No fairness results were provided.",
            "Ensure binary classification + valid sensitive attribute."
        )

    if isinstance(fairness_result, dict) and "error" in fairness_result:
        return (
            "Fairness evaluation failed.",
            fairness_result["error"],
            "Use binary target and a valid sensitive attribute."
        )

    parity_data = None

    if isinstance(fairness_result, dict):
        parity_data = fairness_result.get("Statistical Parity (Selection Rate)", None)

    if parity_data is None:
        return (
            "Fairness evaluation not available.",
            "Statistical parity results missing.",
            "Fairness check requires binary predictions and multiple groups."
        )

    if isinstance(parity_data, dict) and "error" in parity_data:
        return (
            "Fairness check not applicable.",
            parity_data["error"],
            "Use binary classification datasets for fairness checks."
        )

    diff = parity_data.get("difference", None)

    if diff is None:
        return (
            "Fairness check inconclusive.",
            "Group selection rate difference could not be computed.",
            "Ensure at least 2 sensitive groups exist."
        )

    if diff > 0.2:
        return (
            "Fairness risk detected.",
            f"Selection rate difference is high ({round(diff, 3)}).",
            "Apply bias mitigation strategies (reweighting, resampling, threshold tuning)."
        )
    else:
        return (
            "Fairness acceptable.",
            f"Selection rate difference is low ({round(diff, 3)}).",
            "Continue fairness monitoring post-deployment."
        )


def summarize_bias_table(bias_table):
    """
    Converts bias table into short summary instead of printing full table.
    """
    if bias_table is None:
        return ["Bias summary could not be generated (bias table missing)."]

    if hasattr(bias_table, "empty") and bias_table.empty:
        return ["Bias summary could not be generated (bias table empty)."]

    summary_lines = []

    try:
        # Find the group with highest disparity
        max_disparity = 0
        worst_group = None

        for idx in bias_table.index:
            row = bias_table.loc[idx].values
            disparity = max(row) - min(row)

            if disparity > max_disparity:
                max_disparity = disparity
                worst_group = idx

        summary_lines.append(f"Highest bias disparity observed in group: {worst_group}")
        summary_lines.append(f"Maximum disparity across target labels: {round(max_disparity, 2)}%")

        if max_disparity > 20:
            summary_lines.append("Bias Risk: High")
        elif max_disparity > 10:
            summary_lines.append("Bias Risk: Medium")
        else:
            summary_lines.append("Bias Risk: Low")

    except Exception:
        summary_lines.append("Bias summary could not be computed due to formatting issues.")

    return summary_lines


def generate_pdf_report(
    path,
    dataset_quality,
    quality_meta,
    model_metrics,
    fairness_result,
    bias_table=None
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_auto_page_break(auto=True, margin=15)

    # ================= Title =================
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, clean_text("Responsible AI Audit Report"), ln=True)

    pdf.ln(5)

    # ================= Dataset Section =================
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("1. Dataset Assessment"), ln=True)

    status, risk = verdict_from_score(dataset_quality)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 8,
        clean_text(
            f"""
Dataset Quality Score: {dataset_quality}/100
Verdict: {status}
Risk Level: {risk}

Conclusion:
{quality_meta.get("suitability", "Not available")}
"""
        )
    )

    recs = quality_meta.get("recommendations", [])
    if recs:
        pdf.multi_cell(0, 8, clean_text("Recommended Actions:"))
        for r in recs[:5]:  # show only top 5
            pdf.multi_cell(0, 8, clean_text(f"- {r}"))

    # ================= Model Section =================
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("2. Model Performance Assessment"), ln=True)

    perf, reason, action = model_verdict(model_metrics)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 8,
        clean_text(
            f"""
Performance Verdict:
{perf}

Reason:
{reason}

Action:
{action}
"""
        )
    )

    # Print metrics summary (only important)
    pdf.multi_cell(0, 8, clean_text("Key Model Metrics:"))

    if model_metrics.get("task") == "classification":
        pdf.multi_cell(0, 8, clean_text(f"- Accuracy: {model_metrics.get('accuracy', 'N/A')}"))
        pdf.multi_cell(0, 8, clean_text(f"- Precision: {model_metrics.get('precision', 'N/A')}"))
        pdf.multi_cell(0, 8, clean_text(f"- Recall: {model_metrics.get('recall', 'N/A')}"))
        pdf.multi_cell(0, 8, clean_text(f"- F1 Score: {model_metrics.get('f1_score', 'N/A')}"))

    elif model_metrics.get("task") == "regression":
        pdf.multi_cell(0, 8, clean_text(f"- MAE: {model_metrics.get('mae', 'N/A')}"))
        pdf.multi_cell(0, 8, clean_text(f"- RMSE: {model_metrics.get('rmse', 'N/A')}"))

    else:
        pdf.multi_cell(0, 8, clean_text("- Metrics not available."))

    # ================= Fairness Section =================
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("3. Fairness & Bias Assessment"), ln=True)

    fairness, reason, action = fairness_verdict(fairness_result)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 8,
        clean_text(
            f"""
Fairness Verdict:
{fairness}

Reason:
{reason}

Action:
{action}
"""
        )
    )

    # Bias summary short
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, clean_text("Bias Summary (Sensitive Attribute vs Target)"), ln=True)

    pdf.set_font("Arial", size=12)
    bias_summary = summarize_bias_table(bias_table)

    for line in bias_summary:
        pdf.multi_cell(0, 8, clean_text(f"- {line}"))

    # ================= Final Decision =================
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("4. Final Audit Decision"), ln=True)

    model_task = model_metrics.get("task", "unknown")

    # Default final decision
    final_decision = "AUDIT INCOMPLETE"
    final_note = "Audit could not be completed due to missing evaluation results."

    # Classification logic
    if model_task == "classification":
        acc = model_metrics.get("accuracy", None)

        if acc is not None:
            if dataset_quality < 60 or acc < 0.5:
                final_decision = "AUDIT FAILED"
                final_note = "This system is not safe for deployment."
            elif dataset_quality < 80 or acc < 0.7:
                final_decision = "AUDIT CONDITIONAL PASS"
                final_note = "Deploy only after improvements and monitoring."
            else:
                final_decision = "AUDIT PASSED"
                final_note = "System appears deployable with continuous monitoring."

    # Regression logic
    elif model_task == "regression":
        rmse = model_metrics.get("rmse", None)

        if rmse is not None:
            if dataset_quality < 60 or rmse > 50:
                final_decision = "AUDIT FAILED"
                final_note = "Regression model errors are too high for deployment."
            elif dataset_quality < 80 or rmse > 20:
                final_decision = "AUDIT CONDITIONAL PASS"
                final_note = "Deploy only after tuning and evaluation improvements."
            else:
                final_decision = "AUDIT PASSED"
                final_note = "Regression model seems deployable with monitoring."

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 8,
        clean_text(
            f"""
Audit Result: {final_decision}

Final Recommendation:
{final_note}
"""
        )
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf.output(path)

    return path
