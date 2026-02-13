# import pandas as pd

# COLUMNS = [
#     "age","workclass","fnlwgt","education","education-num",
#     "marital-status","occupation","relationship","race",
#     "sex","capital-gain","capital-loss","hours-per-week",
#     "native-country","income"
# ]

# def load_data(path):
#     df = pd.read_csv(path, names=COLUMNS, skipinitialspace=True)
#     df = df.replace("?", pd.NA).dropna()

#     df["income"] = df["income"].apply(
#         lambda x: 1 if ">50K" in x else 0
#     )

#     # Gender
#     df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

#     # Race (White vs Non-White)
#     df["race_binary"] = df["race"].apply(
#         lambda x: 1 if x == "White" else 0
#     )

#     # Age group (>=40 privileged)
#     df["age_group"] = df["age"].apply(
#         lambda x: 1 if x >= 40 else 0
#     )

#     return df




import pandas as pd
import numpy as np

def load_dataset(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_csv(uploaded_file, delimiter=r"\s+", engine="python")

    # Replace common missing indicators
    df.replace(["?", "NA", "N/A", "null", "None", ""], np.nan, inplace=True)

    # Strip spaces in string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df
