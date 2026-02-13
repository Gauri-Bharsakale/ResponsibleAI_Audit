from src.data_loader import load_data
from src.model_training import train_model
from src.explainability import shap_explain
from sklearn.model_selection import train_test_split

df = load_data("data/adult.data")

X = df.drop("income", axis=1)
y = df["income"]
X = X.select_dtypes(include=["int64", "float64"])

X_train, X_test, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)
shap_values = shap_explain(model, X_train, X_test[:100])

print("SHAP explanation generated")
