import pandas as pd


def group_accuracy(df, sensitive_col, y_true_col, y_pred_col):
    """
    Computes accuracy separately for each sensitive group.
    """

    if sensitive_col not in df.columns:
        return {"error": "Sensitive column not found"}

    if y_true_col not in df.columns:
        return {"error": "True label column not found"}

    if y_pred_col not in df.columns:
        return {"error": "Prediction column not found"}

    result = {}
    groups = df[sensitive_col].dropna().unique()

    for g in groups:
        subset = df[df[sensitive_col] == g]

        if len(subset) == 0:
            continue

        acc = (subset[y_true_col] == subset[y_pred_col]).mean()

        result[str(g)] = {
            "accuracy": round(float(acc), 3),
            "samples": int(len(subset))
        }

    return result


def statistical_parity(df, sensitive_col, y_pred_col):
    """
    Measures selection rate difference across sensitive groups.
    Only works for binary classification predictions.
    """

    if sensitive_col not in df.columns:
        return {"error": "Sensitive column not found"}

    if y_pred_col not in df.columns:
        return {"error": "Prediction column not found"}

    df = df.dropna(subset=[sensitive_col, y_pred_col])

    unique_preds = df[y_pred_col].unique()

    if len(unique_preds) != 2:
        return {
            "error": "Statistical parity requires binary predictions",
            "unique_predictions_found": list(unique_preds)
        }

    # Positive class = highest value (works for 0/1 or 1/2)
    positive_class = sorted(unique_preds)[-1]

    selection_rates = {}
    group_sizes = {}

    groups = df[sensitive_col].unique()

    for g in groups:
        subset = df[df[sensitive_col] == g]

        if len(subset) == 0:
            continue

        rate = (subset[y_pred_col] == positive_class).mean()

        selection_rates[str(g)] = round(float(rate), 3)
        group_sizes[str(g)] = int(len(subset))

    if len(selection_rates) < 2:
        return {
            "error": "Not enough groups for fairness comparison",
            "selection_rates": selection_rates,
            "group_sizes": group_sizes
        }

    max_rate = max(selection_rates.values())
    min_rate = min(selection_rates.values())
    difference = max_rate - min_rate

    return {
        "positive_class_used": str(positive_class),
        "selection_rates": selection_rates,
        "group_sizes": group_sizes,
        "max_rate": round(max_rate, 3),
        "min_rate": round(min_rate, 3),
        "difference": round(difference, 3)
    }
