# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
import pandas as pd
import numpy as np
from preprocess.preprocess import StratifiedSplitter, PartialStandardizer
from models.models import  LogisticRegression
from evaluation.metrics import ConfusionEvaluator, confusion_stats
from evaluation.threshold import FScoreThresholdAnalyzer, plot_confusion_matrix
# %%
# 讀取資料
df = pd.read_csv('../data/diabetes_binary_health_indicators_BRFSS2015.csv')

# 預設trian:val:test = 6:2:2, seed = 42
splitter = StratifiedSplitter(df)
train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y, train_idx, val_idx, test_idx = splitter.split()

# 標準化
feature_names = df.drop(columns=['Diabetes_binary']).columns.tolist()

scaler = PartialStandardizer(feature_names)
scaler.fit(train_val_x)
train_x_scaled = scaler.transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)

# %%
# Logistic Regression with Gradient Descent: 跑很多次以後發現lr0.01是可以的
# epoch在有設定early stopping的情況下大概都跑9000結束
logistic_model = LogisticRegression(lr=0.01, epoch=10000, verbose=True)
logistic_model.fit(train_x_scaled, train_y, X_val=val_x_scaled, y_val=val_y)

# 畫出loss
logistic_model.plot_loss()

# %%
# 進行預測
y_score = logistic_model.predict_proba(test_x_scaled)

# %%
# 找出在Val set裡面F2 Score最高的threshold
ConfusionEvaluator.find_best_threshold_fbeta(logistic_model, val_x_scaled, val_y, beta=2.0)  # F2
y_score_val = logistic_model.predict_proba(val_x_scaled)
analyzer_val = FScoreThresholdAnalyzer(val_y, y_score_val)
analyzer_val.compute()
analyzer_val.plot()
best_thresholds = analyzer_val.get_best_thresholds()
best_thresholds['best_threshold_f2']

# %%
# 把前面得到的最佳threshold套用到test set
y_score_test = logistic_model.predict_proba(test_x_scaled)

# 用自定義的evaluator顯示test set最後的指標
ConfusionEvaluator.evaluate(test_y, y_score_test, threshold=best_thresholds['best_threshold_f2'])
print('-'*20)

# 算出confusion matrix需要之數字
y_pred_test = (y_score_test >= best_thresholds['best_threshold_f2']).astype(int)
TP, FP, TN, FN = confusion_stats(test_y, y_pred_test)

# 畫出confusion matrix
plot_confusion_matrix(TP, FP, TN, FN)

# %% [markdown]
# ===Undersampling===

# %%
# 定義 1:1 隨機抽樣函數，將資料中的正負樣本均衡（undersampling）
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

# %%
# 再次使用 StratifiedSplitter 來切分 train/val/test 資料（此時尚未 undersample）
splitter = StratifiedSplitter(df)  
train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y, train_idx, val_idx, test_idx = splitter.split()

# 對 train/val set 做 undersampling，讓 0:1 數量變成 1:1
train_x_bal, train_y_bal = balance_1to1(train_x, train_y)
val_x, val_y = balance_1to1(val_x, val_y)

# 重新標準化資料（以 undersampled train + val 作為 fit 的依據）
feature_names = df.drop(columns=['Diabetes_binary']).columns.tolist()
scaler = PartialStandardizer(feature_names)
scaler.fit(np.vstack([train_x_bal, val_x]))

train_x_scaled = scaler.transform(train_x_bal)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)

# %%
# 訓練 Logistic Regression (undersampled)
logistic_model_us = LogisticRegression(lr=0.01, epoch=10000, verbose=True)
logistic_model_us.fit(train_x_scaled, train_y_bal, X_val=val_x_scaled, y_val=val_y)

# %%
# 畫出 loss 曲線
y_score = logistic_model_us.predict_proba(test_x_scaled)
logistic_model_us.plot_loss()

# %%
# 找出在 Validation set 上 F2 Score 最佳的 threshold
ConfusionEvaluator.find_best_threshold_fbeta(logistic_model_us, val_x_scaled, val_y, beta=2.0)  # F2

# %%
# 繪製 F-score 隨 threshold 變化的圖形，找出最佳 F2 threshold
y_score_val = logistic_model_us.predict_proba(val_x_scaled)
analyzer_val = FScoreThresholdAnalyzer(val_y, y_score_val)
analyzer_val.compute()
analyzer_val.plot()
best_thresholds = analyzer_val.get_best_thresholds()
best_thresholds['best_threshold_f2']

# %%
# 把前面得到的最佳 threshold 套用到 test set，並計算各項指標
y_score_test = logistic_model_us.predict_proba(test_x_scaled)
ConfusionEvaluator.evaluate(test_y, y_score_test, threshold=best_thresholds['best_threshold_f2'])

# %%
# 計算 confusion matrix 所需的四個指標
y_pred_test = (y_score_test >= best_thresholds['best_threshold_f2']).astype(int)
TP, FP, TN, FN = confusion_stats(test_y, y_pred_test)

# 畫出 confusion matrix
plot_confusion_matrix(TP, FP, TN, FN)




