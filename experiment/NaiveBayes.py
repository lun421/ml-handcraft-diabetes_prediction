# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
import pandas as pd
from preprocess.preprocess import StratifiedSplitter, PartialStandardizer
from models.models import  NaiveBayes
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
# Naive Bayes model
nb_model = NaiveBayes(train_x_scaled, train_y)
nb_model.fit()

# %%
# Find best threshold with F2
best_t_f2 = ConfusionEvaluator.find_best_threshold_fbeta(nb_model, val_x_scaled, val_y, beta=2.0)
print(f'best threshold using  F2: {best_t_f2}')

# %%
# plot F2 score change on different threshold
val_score = nb_model.predict_proba(val_x_scaled)[:, 1]
analyzer = FScoreThresholdAnalyzer(val_y, val_score)
analyzer.compute()
analyzer.plot()
best_thresholds = analyzer.get_best_thresholds()

# %%
# Evaluate Testing Set
y_score_test = nb_model.predict_proba(test_x_scaled)[:, 1]
y_pred_test = (y_score_test >= best_t_f2).astype(int)
ConfusionEvaluator.evaluate(test_y, y_score_test, threshold=best_t_f2)

# %%
# print confusion matrix
TP, FP, TN, FN = confusion_stats(test_y, y_pred_test)
plot_confusion_matrix(TP, FP, TN, FN)
