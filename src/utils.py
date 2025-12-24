import pandas as pd
import numpy as np

def load_data(filepath, target_col='Is_Human', test_size=0.3, seed=42):
    # load csv
    df = pd.read_csv(filepath)
    
    # shuffle
    np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # split features/target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values.reshape(-1, 1)
    
    # manual split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # z-score norm (critical for mlp)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_metrics(y_true, y_pred):
    # manual calc
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return prec, rec