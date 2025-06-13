#%%
import numpy as np

# 基本 Confusion Matrix 計算
def confusion_stats(y_true, y_pred):
    TP = FP = TN = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 1 and pred == 0:
            FN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 0 and pred == 0:
            TN += 1
    return TP, FP, TN, FN

class ConfusionEvaluator:
    @staticmethod
    def evaluate(y_true, y_score, threshold=0.5):
        y_pred = (y_score >= threshold).astype(int)
        TP, FP, TN, FN = confusion_stats(y_true, y_pred)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (precision + recall) != 0 else 0

        print(f"Confusion Matrix:   threshold={threshold:.3f}")
        print(f"[[TN={TN}, FP={FP}]\n [FN={FN}, TP={TP}]]")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"F2 Score:   {f2:.3f}")

        return {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1
        }

    @staticmethod
    def find_best_threshold_fbeta(model, X_val, y_val, beta=2.0):
        best_threshold = 0.0
        best_fbeta = 0.0
        best_stats = None

        try:
            y_score = model.predict_proba(X_val)
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
        except:
            try:
                y_score = model.predict_score(X_val)
            except:
                y_score = model.predict(X_val)

        thresholds = np.linspace(np.min(y_score), np.max(y_score), 1001)

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            TP, FP, TN, FN = confusion_stats(y_val, y_pred)
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0

            if precision + recall == 0:
                fbeta = 0
            else:
                fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = threshold
                best_stats = (TP, FP, TN, FN, precision, recall)

        TP, FP, TN, FN, precision, recall = best_stats
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        print("\n" + "=" * 40)
        print(f"Best Threshold based on F{beta:.1f} Score")
        print("=" * 40)
        print(f"Threshold:      {best_threshold:.3f}")
        print(f"Max F{beta:.1f} Score: {best_fbeta:.4f}")
        print(f"[[TN={TN:5d}, FP={FP:5d}]\n [FN={FN:5d}, TP={TP:5d}]]")
        print(f"  Accuracy:      {(TP + TN) / (TP + TN + FP + FN):.4f}")
        print(f"  Precision:     {precision:.4f}")
        print(f"  Recall:        {recall:.4f}")
        print(f"  F1 Score:      {f1:.4f}")
        print(f"  F{beta:.1f} Score:   {best_fbeta:.4f}")
        print("=" * 40 + "\n")

        return best_threshold
# %%
