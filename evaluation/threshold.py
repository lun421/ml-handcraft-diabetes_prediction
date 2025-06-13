import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(TP, FP, TN, FN, labels=["Negative", "Positive"]):
    cm = np.array([[TN, FP], [FN, TP]])
    group_names = [["TN", "FP"], ["FN", "TP"]]
    annotations = [[f"{name}\n{count}" for name, count in zip(row_names, row)]
                   for row_names, row in zip(group_names, cm)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})

    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


class FScoreThresholdAnalyzer:
    def __init__(self, y_true, y_score):
        self.y_true = np.array(y_true)
        self.y_score = np.array(y_score)
        self.thresholds = np.linspace(0.0, 1.0, 1001)
        self.f1_scores = []
        self.f2_scores = []
        self.best_threshold_f1 = None
        self.best_threshold_f2 = None
        self.max_f1 = None
        self.max_f2 = None

    def compute(self):
        f1s, f2s = [], []
        for t in self.thresholds:
            y_pred = (self.y_score >= t).astype(int)
            TP = np.sum((y_pred == 1) & (self.y_true == 1))
            FP = np.sum((y_pred == 1) & (self.y_true == 0))
            FN = np.sum((y_pred == 0) & (self.y_true == 1))

            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall != 0 else 0

            f1s.append(f1)
            f2s.append(f2)

        self.f1_scores = f1s
        self.f2_scores = f2s
        self.max_f1 = max(f1s)
        self.best_threshold_f1 = self.thresholds[np.argmax(f1s)]
        self.max_f2 = max(f2s)
        self.best_threshold_f2 = self.thresholds[np.argmax(f2s)]

    def plot(self):
        if not self.f1_scores or not self.f2_scores:
            self.compute()

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, self.f1_scores, label="F1 Score", color='orange')
        plt.plot(self.thresholds, self.f2_scores, label="F2 Score", color='blue')

        plt.axvline(self.best_threshold_f1, color='orange', linestyle='--', alpha=0.5,
                    label=f'Best F1 @ {self.best_threshold_f1:.3f}')
        plt.axvline(self.best_threshold_f2, color='blue', linestyle='--', alpha=0.5,
                    label=f'Best F2 @ {self.best_threshold_f2:.3f}')

        plt.text(self.best_threshold_f1 + 0.015, self.max_f1, f'{self.max_f1:.3f}', color='orange',
                 ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
        plt.text(self.best_threshold_f2 - 0.015, self.max_f2, f'{self.max_f2:.3f}', color='blue',
                 ha='right', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold vs F1, F2 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_best_thresholds(self):
        if self.best_threshold_f1 is None or self.best_threshold_f2 is None:
            self.compute()
        return {
            "best_threshold_f1": self.best_threshold_f1,
            "max_f1": self.max_f1,
            "best_threshold_f2": self.best_threshold_f2,
            "max_f2": self.max_f2
        }
