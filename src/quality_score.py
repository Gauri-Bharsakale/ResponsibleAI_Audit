# import numpy as np

# def compute_quality_score(df, missing_report, duplicates, outliers, imbalance=None, high_corr_count=0):
#     score = 100
#     recommendations = []
#     risk_flags = []

#     # Missing values penalty
#     if not missing_report.empty:
#         missing_percent_total = missing_report["missing_percent"].sum()
#         penalty = min(30, missing_percent_total / 2)
#         score -= penalty
#         recommendations.append("Handle missing values (drop, impute mean/median/mode).")
#         risk_flags.append("Missing Values Detected")

#     # Duplicate penalty
#     if duplicates > 0:
#         penalty = min(15, duplicates / len(df) * 100)
#         score -= penalty
#         recommendations.append("Remove duplicate rows to avoid biased training.")
#         risk_flags.append("Duplicate Rows Found")

#     # Outlier penalty
#     if len(outliers) > 0:
#         penalty = min(15, len(outliers) * 2)
#         score -= penalty
#         recommendations.append("Treat outliers using capping, transformation, or removal.")
#         risk_flags.append("Outliers Detected")

#     # Class imbalance penalty
#     if imbalance is not None:
#         if len(imbalance) > 1:
#             min_class = imbalance.min()
#             if min_class < 20:
#                 score -= 20
#                 recommendations.append("Fix class imbalance using SMOTE / undersampling / class weights.")
#                 risk_flags.append("Target Imbalance Risk")

#     # Correlation leakage penalty
#     if high_corr_count > 0:
#         penalty = min(20, high_corr_count * 5)
#         score -= penalty
#         recommendations.append("Remove highly correlated features to avoid leakage.")
#         risk_flags.append("High Correlation Features")

#     # Small dataset penalty
#     if len(df) < 1000:
#         score -= 15
#         recommendations.append("Dataset is small; consider collecting more samples.")
#         risk_flags.append("Low Sample Size")

#     # Clamp score
#     score = max(0, min(100, score))

#     # Risk level
#     if score >= 80:
#         risk_level = "LOW"
#         suitability = "✅ Suitable for ML Training"
#     elif score >= 50:
#         risk_level = "MEDIUM"
#         suitability = "⚠️ Usable but Needs Cleaning"
#     else:
#         risk_level = "HIGH"
#         suitability = "❌ Not Recommended Without Fixes"

#     return {
#         "quality_score": round(score, 2),
#         "risk_level": risk_level,
#         "suitability": suitability,
#         "recommendations": recommendations,
#         "risk_flags": risk_flags
#     }








def compute_quality_score(df, missing_report, duplicates, outliers, imbalance):
    score = 100
    recommendations = []

    missing_percent = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

    if missing_percent > 5:
        score -= 20
        recommendations.append("Handle missing values (imputation or drop missing rows).")

    if duplicates > 0:
        score -= 10
        recommendations.append("Remove duplicate rows.")

    total_outliers = sum(outliers.values())
    if total_outliers > 0:
        score -= 10
        recommendations.append("Handle outliers using capping/removal/transformations.")

    if len(imbalance) > 0:
        max_class = max(imbalance.values())
        if max_class > 80:
            score -= 20
            recommendations.append("Dataset is highly imbalanced. Use SMOTE or class weighting.")

    if score >= 80:
        risk = "Low Risk"
        suitability = "Dataset is suitable for ML usage."
    elif score >= 60:
        risk = "Medium Risk"
        suitability = "Dataset can be used but needs cleaning."
    else:
        risk = "High Risk"
        suitability = "Dataset is not reliable unless fixed."

    return {
        "quality_score": max(score, 0),
        "risk_level": risk,
        "suitability": suitability,
        "recommendations": recommendations
    }
