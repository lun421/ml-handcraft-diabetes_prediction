import numpy as np

def balance_1to1(X, y, seed=42):
    np.random.seed(seed)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    n = min(len(idx_0), len(idx_1))
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    idx_bal = np.concatenate([idx_0[:n], idx_1[:n]])
    np.random.shuffle(idx_bal)
    return X[idx_bal], y[idx_bal]
