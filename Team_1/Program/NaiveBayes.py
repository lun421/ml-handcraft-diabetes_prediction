# %%
import pandas as pd

from Preprocess import StratifiedSplitter, PartialStandardizer
from Models import  ConfusionEvaluator, NaiveBayes, FScoreThresholdAnalyzer, confusion_stats, plot_confusion_matrix

# %%
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# 預設trian:val:test = 6:2:2, seed = 42
splitter = StratifiedSplitter(df)
# train_val_x, train_val_y 代表train val都有，用於不需要validation的情況
train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y, train_idx, val_idx, test_idx = splitter.split()

# 標準化：只對數值型資料進行標準化
feature_names = df.drop(columns=['Diabetes_binary']).columns.tolist()
scaler = PartialStandardizer(feature_names)
scaler.fit(train_val_x)
train_x_scaled = scaler.transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)

# %%
# 套用Naive Bayes model
nb_model = NaiveBayes(train_x_scaled, train_y)
nb_model.fit()

# %%
# 使用validation set找到F2分數最佳的threshold
best_t_f2 = ConfusionEvaluator.find_best_threshold_fbeta(nb_model, val_x_scaled, val_y, beta=2.0)
print(f'best threshold using  F2: {best_t_f2}')

# %%
# Evaluate Testing Set：使用val跑出來最佳的threshold進行測驗
y_score_test = nb_model.predict_proba(test_x_scaled)[:, 1]
y_pred_test = (y_score_test >= best_t_f2).astype(int)
ConfusionEvaluator.evaluate(test_y, y_score_test, threshold=best_t_f2)

# %%
# print confusion matrix
TP, FP, TN, FN = confusion_stats(test_y, y_pred_test)
plot_confusion_matrix(TP, FP, TN, FN)

# %%
# plot 不同 threshold 切分導致的F score變化 (validation set)
val_score = nb_model.predict_proba(val_x_scaled)[:, 1]
analyzer = FScoreThresholdAnalyzer(val_y, val_score)
analyzer.compute()
analyzer.plot()
best_thresholds = analyzer.get_best_thresholds()

# %%
# plot 不同 threshold 切分導致的F score變化 (test set)
analyzer = FScoreThresholdAnalyzer(test_y, y_score_test)
analyzer.compute()
analyzer.plot()
best_thresholds = analyzer.get_best_thresholds()

