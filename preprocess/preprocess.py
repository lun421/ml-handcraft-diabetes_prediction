# %%
import numpy as np

class StratifiedSplitter:
    """
    預設 train:val:test = 6:2:2
    使用方法：
    splitter = StratifiedSplitter(df)
    train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y = splitter.split()
    """

    # 預設train:val:test = 6:2:2
    def __init__(self, data, label_col='Diabetes_binary', train_ratio=0.6, val_ratio=0.2, seed=42):
        self.data = data.reset_index(drop=True) 
        self.label_col = label_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self.x = data.drop(label_col, axis=1).values
        self.y = data[label_col].values

    # 進行切分
    def split(self):
        np.random.seed(self.seed)
        y = self.y
        x = self.x

        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        np.random.shuffle(idx_0)
        np.random.shuffle(idx_1)

        # 對label=0 or 1的資料分別進行6:2:2的切分
        n0, n1 = len(idx_0), len(idx_1)
        train_0 = int(self.train_ratio * n0)
        val_0 = int(self.val_ratio * n0)
        train_1 = int(self.train_ratio * n1)
        val_1 = int(self.val_ratio * n1)

        #把切分完成的不同label合併就能得到完整的set
        train_idx = np.concatenate([idx_0[:train_0], idx_1[:train_1]])
        val_idx = np.concatenate([idx_0[train_0:train_0+val_0], idx_1[train_1:train_1+val_1]])
        test_idx = np.concatenate([idx_0[train_0+val_0:], idx_1[train_1+val_1:]])

        # 再進行一次shuffle確保隨機
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        train_x = x[train_idx]
        train_y = y[train_idx]
        val_x = x[val_idx]
        val_y = y[val_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
        train_val_x = np.concatenate([train_x, val_x])
        train_val_y = np.concatenate([train_y, val_y])

        return (train_x, train_y, val_x, val_y, test_x, test_y,
                train_val_x, train_val_y,
                train_idx, val_idx, test_idx)


#%%
class PartialStandardizer:
    """
    使用方法：
    scaler = PartialStandardizer(feature_names)
    scaler.fit(train_val_x)

    train_x_scaled = scaler.transform(train_x)
    val_x_scaled = scaler.transform(val_x)
    test_x_scaled = scaler.transform(test_x)
    train_val_x_scaled = scaler.transform(train_val_x)
    """
    binary_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                   'DiffWalk', 'Sex']
    
    # 不對binary的類別進行，但對數字類別進行標準化
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.numerical_indices = [i for i, col in enumerate(feature_names) if col not in self.binary_cols]
        self.mean = None
        self.std = None

    # 根據提供的資料計算 mean 和 std，只針對 numeric 欄位
    def fit(self, x):
        self.mean = x[:, self.numerical_indices].mean(axis=0)
        self.std = x[:, self.numerical_indices].std(axis=0)
        return self

    # 套用標準化
    def transform(self, x):
        x_copy = x.copy()
        x_copy[:, self.numerical_indices] = (x[:, self.numerical_indices] - self.mean) / self.std
        return x_copy

    # 同時進行 fit 和 transform的比較快的用法
    def fit_transform(self, x):
        return self.fit(x).transform(x)