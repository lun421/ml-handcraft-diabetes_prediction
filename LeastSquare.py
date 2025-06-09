# %%
import pandas as pd
import numpy as np
from Preprocess import StratifiedSplitter, PartialStandardizer
from Models import confusion_stats, ConfusionEvaluator, plot_confusion_matrix, FScoreThresholdAnalyzer


# %%
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
# 預設trian:val:test = 6:2:2, seed = 42
splitter = StratifiedSplitter(df)
# train_val_x, train_val_y 代表train val都有，對於不用分val的模型會有用
train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y, train_idx, val_idx, test_idx = splitter.split()


#標準化
feature_names = df.drop(columns=['Diabetes_binary']).columns.tolist()
scaler = PartialStandardizer(feature_names)
scaler.fit(train_val_x)
train_x_scaled = scaler.transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)

# %%
#用ordinary least square找到向量的參數，也就是各個feature的係數
XtX = np.dot(train_x_scaled.T, train_x_scaled)
Xty = np.dot(train_x_scaled.T, train_y)
theta_ls = np.dot(np.linalg.inv(XtX), Xty)

# %%
#用前面找到的係數theta_ls來對validation set做回歸
y_val_score = np.dot(val_x_scaled,theta_ls)

#然後用get_best_threshold跑loop找到f2最高的threshold
analyzer = FScoreThresholdAnalyzer(val_y, y_val_score)
analyzer.compute()
best = analyzer.get_best_thresholds()

# %%
#對test一樣用theta_ls去計算回歸，然後用剛剛val set找到的best threshold作為正式預測用的threshold
y_test_score = np.dot(test_x_scaled,theta_ls)
y_test_pred = (y_test_score >= best['best_threshold_f2']).astype(int)

# %%
#繪製confusion matrix和打印各項指標
TP, FP, TN, FN = confusion_stats(test_y, y_test_pred)
plot_confusion_matrix(TP, FP, TN, FN)
ConfusionEvaluator.evaluate(test_y, y_test_score, threshold=best['best_threshold_f2'])


#%%
# 視覺化F1和F2分數隨著threshold變化的情況
analyzer_test = FScoreThresholdAnalyzer(test_y, y_test_score)
analyzer_test.compute()
analyzer_test.plot()