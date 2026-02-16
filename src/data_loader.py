import pandas as pd

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


def load_dataset(uploaded_file):
    """
    Loads dataset uploaded by user (CSV/TXT/DATA).
    Works properly in Streamlit deployment.
    """

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        return df

    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=r"\s+", engine="python")
            return df

        except Exception:
            uploaded_file.seek(0)
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
