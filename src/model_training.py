# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# def train_model(X, y):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X, y)
#     return model







from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np


def detect_problem_type(df, target_col):
    if df[target_col].dtype in ["int64", "float64"]:
        if df[target_col].nunique() > 20:
            return "regression"
    return "classification"


def train_model(df, target_col):
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = detect_problem_type(df, target_col)

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    if problem_type == "classification":
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000))
        ])
    else:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test, problem_type
