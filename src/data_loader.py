import pandas as pd


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


def load_dataset(uploaded_file):
    """
    Loads dataset uploaded by user (CSV/TXT/DATA).
    """

    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        try:
            df = pd.read_csv(uploaded_file, delimiter=r"\s+", engine="python")
            return df
        except Exception:
            df = pd.read_csv(uploaded_file, header=None)
            return df


def load_demo_dataset():
    """
    Loads Adult Income dataset from data/adult.data
    """

    path = "data/adult.data"

    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True
    )

    return df
