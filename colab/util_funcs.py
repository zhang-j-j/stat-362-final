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

def sentence_embeddings(data, model):
    """
    Generate sentence embeddings for baseline models.

    Args:
        data (tuple): Tuple containing training, validation, and test data
        model: SentenceTransformer model

    Returns:
        X_train_emb: Training embeddings
        X_val_emb: Validation embeddings
        X_test_emb: Test embeddings
    """
    X_train, X_val, X_test = data
    X_train_emb = model.encode(X_train, show_progress_bar=True, convert_to_numpy=True)
    X_val_emb = model.encode(X_val, show_progress_bar=True, convert_to_numpy=True)
    X_test_emb = model.encode(X_test, show_progress_bar=True, convert_to_numpy=True)

    print(f"Training embeddings shape: {X_train_emb.shape}")
    print(f"Validation embeddings shape: {X_val_emb.shape}")
    print(f"Test embeddings shape: {X_test_emb.shape}")
    return X_train_emb, X_val_emb, X_test_emb
