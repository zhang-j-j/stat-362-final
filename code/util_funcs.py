import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_split_data(path, random_state=42):
    """
    Load data and split into training, validation, and testing sets with a 60/20/20 ratio.

    Parameters:
    path (str): File path
    random_state (int): Random state for reproducibility

    Returns:
    X_train (DataFrame): Training features
    X_val (DataFrame): Validation features
    X_test (DataFrame): Testing features
    y_train (Series): Training labels
    y_val (Series): Validation labels
    y_test (Series): Testing labels
    """
    df = pd.read_csv(path)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    # first split (train + val, test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['content'].values,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    # second split (train, val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

