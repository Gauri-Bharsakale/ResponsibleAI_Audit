# import pandas as pd
# import numpy as np

# def dataset_summary(df):
#     return {
#         "rows": df.shape[0],
#         "columns": df.shape[1],
#         "column_names": list(df.columns),
#         "dtypes": df.dtypes.astype(str).to_dict()
#     }

# def missing_value_report(df):
#     missing_count = df.isnull().sum()
#     missing_percent = (missing_count / len(df)) * 100

#     report = pd.DataFrame({
#         "missing_count": missing_count,
#         "missing_percent": missing_percent
#     })

#     report = report[report["missing_count"] > 0].sort_values(
#         by="missing_percent", ascending=False
#     )

#     return report

# def duplicate_report(df):
#     return df.duplicated().sum()

# def detect_outliers_iqr(df):
#     outlier_info = {}

#     numeric_cols = df.select_dtypes(include=[np.number]).columns

#     for col in numeric_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1

#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR

#         outliers = df[(df[col] < lower) | (df[col] > upper)]

#         if len(outliers) > 0:
#             outlier_info[col] = len(outliers)

#     return outlier_info

# def class_imbalance(df, target_col):
#     if target_col not in df.columns:
#         return None

#     distribution = df[target_col].value_counts(normalize=True) * 100
#     return distribution








import pandas as pd
import numpy as np

def dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def missing_value_report(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing.reset_index().rename(columns={"index": "Column", 0: "Missing_Count"})

def duplicate_report(df):
    return df.duplicated().sum()

def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    outlier_report = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_report[col] = len(outliers)

    return outlier_report

def class_imbalance(df, target_col):
    counts = df[target_col].value_counts(normalize=True) * 100
    return counts.to_dict()


def validate_target_column(df, target_col):
    if target_col not in df.columns:
        return False, "Target column not found."

    unique_vals = df[target_col].nunique()

    if unique_vals <= 1:
        return False, "Target column has only one unique value."

    # ID-like detection
    if unique_vals > 0.9 * len(df):
        return False, "Target column looks like an ID (too many unique values)."

    return True, "Target column looks valid."
