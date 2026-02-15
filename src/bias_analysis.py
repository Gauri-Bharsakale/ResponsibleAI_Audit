import pandas as pd


def check_bias_distribution(df, sensitive_col, target_col):
    """
    Creates a bias distribution table:
    Shows percentage distribution of target values across sensitive groups.

    Example:
    Gender (Male/Female) vs Income (>50K / <=50K)
    Output in % format.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if sensitive_col not in df.columns:
        return pd.DataFrame({"error": [f"Sensitive column '{sensitive_col}' not found"]})

    if target_col not in df.columns:
        return pd.DataFrame({"error": [f"Target column '{target_col}' not found"]})

    # Drop missing values only for required columns
    df_clean = df.dropna(subset=[sensitive_col, target_col])

    if df_clean.empty:
        return pd.DataFrame({"error": ["Dataset is empty after removing missing sensitive/target values"]})

    # Create Crosstab distribution
    bias_table = pd.crosstab(
        df_clean[sensitive_col],
        df_clean[target_col],
        normalize="index"
    ) * 100

    bias_table = bias_table.round(2)

    # Add row count for each sensitive group
    group_counts = df_clean[sensitive_col].value_counts()
    bias_table.insert(0, "Group_Count", group_counts)

    return bias_table


def bias_summary(bias_table):
    """
    Generates a short bias summary from the bias_table.
    Useful for report generator / Streamlit display.
    """

    if bias_table is None or bias_table.empty:
        return {
            "status": "Bias summary not available",
            "max_disparity": None,
            "worst_group": None,
            "risk": "Unknown"
        }

    if "Group_Count" in bias_table.columns:
        table_only = bias_table.drop(columns=["Group_Count"])
    else:
        table_only = bias_table

    max_disparity = 0
    worst_group = None

    for idx in table_only.index:
        values = table_only.loc[idx].values

        disparity = max(values) - min(values)

        if disparity > max_disparity:
            max_disparity = disparity
            worst_group = idx

    # Risk classification (percentage based)
    if max_disparity > 20:
        risk = "High"
    elif max_disparity > 10:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "status": "Bias summary generated",
        "max_disparity": round(float(max_disparity), 2),
        "worst_group": str(worst_group),
        "risk": risk
    }
