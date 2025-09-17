# utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, sheet_name="Sheet1"):
    """Load Excel sheet as DataFrame."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(f"[INFO] Loaded data shape: {df.shape}")
    return df

def split_data(df, features, targets, test_size=0.25):
    """Split DataFrame into X_train, X_test, y_train, y_test."""
    X = df[features].values
    y = df[targets].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"[INFO] Split: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test
