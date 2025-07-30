import pandas as pd

def split_features_labels(df: pd.DataFrame):
    """Splits features (X) and target labels (y) from the dataset."""
    X = df.drop(columns=["Species"])
    y = df["Species"]
    return X, y
