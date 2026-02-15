from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression


def detect_problem_type(df, target_col):
    """
    Detect whether task is classification or regression.
    - If target has small number of unique values -> classification
    - Else if target is numeric and many unique values -> regression
    """

    unique_vals = df[target_col].nunique()

    # Common classification case: numeric labels like 0/1 or 1/2/3
    if unique_vals <= 10:
        return "classification"

    # Regression case
    if df[target_col].dtype in ["int64", "float64"] and unique_vals > 20:
        return "regression"

    return "classification"


def train_model(df, target_col):
    """
    Trains a baseline model with preprocessing pipeline.
    Returns: model, X_test, y_test, problem_type
    """

    # Drop rows only where target is missing
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = detect_problem_type(df, target_col)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols)
        ]
    )

    # Choose model based on problem type
    if problem_type == "classification":
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000))
        ])
    else:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if problem_type == "classification" and y.nunique() <= 20 else None
    )

    # Fit model
    model.fit(X_train, y_train)

    return model, X_test, y_test, problem_type
