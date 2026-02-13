# from aif360.metrics import ClassificationMetric

# def compute_fairness_metrics(dataset_true, dataset_pred,
#                              privileged, unprivileged):

#     metric = ClassificationMetric(
#         dataset_true,
#         dataset_pred,
#         privileged_groups=privileged,
#         unprivileged_groups=unprivileged
#     )

#     return {
#         "statistical_parity": metric.statistical_parity_difference(),
#         "equal_opportunity": metric.equal_opportunity_difference(),
#         "false_positive_rate": metric.false_positive_rate_difference()
#     }








import pandas as pd


def group_accuracy(df, sensitive_col, y_true_col, y_pred_col):
    result = {}

    if sensitive_col not in df.columns:
        return {"error": "Sensitive column not found"}

    groups = df[sensitive_col].dropna().unique()

    for g in groups:
        subset = df[df[sensitive_col] == g]

        if len(subset) == 0:
            continue

        acc = (subset[y_true_col] == subset[y_pred_col]).mean()
        result[str(g)] = round(float(acc), 3)

    return result


def statistical_parity(df, sensitive_col, y_pred_col):
    """
    Measures selection rate difference across sensitive groups.
    Works only when prediction is binary (0/1 or 2 unique values).
    """

    if sensitive_col not in df.columns:
        return {"error": "Sensitive column not found"}

    if y_pred_col not in df.columns:
        return {"error": "Prediction column not found"}

    unique_preds = df[y_pred_col].dropna().unique()

    if len(unique_preds) != 2:
        return {"error": "Statistical parity requires binary predictions"}

    positive_class = sorted(unique_preds)[-1]  # treat highest value as positive

    result = {}
    groups = df[sensitive_col].dropna().unique()

    for g in groups:
        subset = df[df[sensitive_col] == g]

        if len(subset) == 0:
            continue

        selection_rate = (subset[y_pred_col] == positive_class).mean()
        result[str(g)] = round(float(selection_rate), 3)

    return {
        "selection_rates": result,
        "max_rate": round(max(result.values()), 3) if result else None,
        "min_rate": round(min(result.values()), 3) if result else None,
        "difference": round(max(result.values()) - min(result.values()), 3) if len(result) > 1 else None
    }
