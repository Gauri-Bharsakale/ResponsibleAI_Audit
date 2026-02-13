# def representation_bias(df, attribute):
#     return df[attribute].value_counts(normalize=True)

# def label_bias(df, attribute, label="income"):
#     return df.groupby(attribute)[label].mean()







import pandas as pd

def check_bias_distribution(df, sensitive_col, target_col):
    """
    Shows distribution of target values across sensitive attribute.
    """
    bias_table = pd.crosstab(df[sensitive_col], df[target_col], normalize="index") * 100
    return bias_table.round(2)
