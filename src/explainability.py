# import shap

# def shap_explain(model, X_train, X_test):
#     explainer = shap.Explainer(model, X_train)
#     shap_values = explainer(X_test)
#     return shap_values






import shap
import numpy as np
import pandas as pd

def generate_shap_values(model, X_sample):
    """
    Generate SHAP values for explainability.
    """
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    return shap_values


def get_top_features(model, top_n=10):
    try:
        classifier = model.named_steps.get("classifier", None)
        if classifier is None:
            return None

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        coefs = classifier.coef_[0]

        importance = np.abs(coefs)
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

        return df_imp.head(top_n)

    except:
        return None
