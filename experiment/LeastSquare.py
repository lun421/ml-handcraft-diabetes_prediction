# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
import pandas as pd
import numpy as np
from preprocess.preprocess import StratifiedSplitter, PartialStandardizer
from models.models import  LeastSquaresClassifier
from evaluation.metrics import ConfusionEvaluator, confusion_stats
from evaluation.threshold import FScoreThresholdAnalyzer, plot_confusion_matrix
# %%
# load data
df = pd.read_csv('../data/diabetes_binary_health_indicators_BRFSS2015.csv')
# trian:val:test = 6:2:2, seed = 42
splitter = StratifiedSplitter(df)
train_x, train_y, val_x, val_y, test_x, test_y, train_val_x, train_val_y, train_idx, val_idx, test_idx = splitter.split()


# Standardize
feature_names = df.drop(columns=['Diabetes_binary']).columns.tolist()
scaler = PartialStandardizer(feature_names)
scaler.fit(train_val_x)
train_x_scaled = scaler.transform(train_x)
val_x_scaled = scaler.transform(val_x)
test_x_scaled = scaler.transform(test_x)

# %%
# OLS model
model = LeastSquaresClassifier()
model.fit(train_x_scaled, train_y)

# %%
# threshold optimize
y_val_score = model.predict_score(val_x_scaled)
analyzer = FScoreThresholdAnalyzer(val_y, y_val_score)
analyzer.compute()
best = analyzer.get_best_thresholds()
analyzer.plot()


# %%
# Evaluate Testing Set
y_test_score = model.predict_score(test_x_scaled)
y_test_pred = model.predict(test_x_scaled, threshold=best['best_threshold_f2'])


# %%
# print confusion matrix
TP, FP, TN, FN = confusion_stats(test_y, y_test_pred)
plot_confusion_matrix(TP, FP, TN, FN)
ConfusionEvaluator.evaluate(test_y, y_test_score, threshold=best['best_threshold_f2'])


