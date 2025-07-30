import pandas as pd

def split_features_labels(df: pd.DataFrame):
    """Splits features (X) and target labels (y) from the dataset."""
    X = df.drop(columns=["Species"])
    y = df["Species"]
    return X, y


from sklearn.model_selection import train_test_split

def train_test_data(X, y, test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
