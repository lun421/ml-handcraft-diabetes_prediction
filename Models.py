import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

# Confusion Matrix
# 基本的confunsion matrix計算
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


# 通用 Confusion Evaluator
# ----------------------------------
class ConfusionEvaluator:
    @staticmethod
    def evaluate(y_true, y_score, threshold=0.5):
        y_pred = (y_score >= threshold).astype(int)

        # 運用上面做的confusion matrixc函示來算，差別是有套用threshold
        TP, FP, TN, FN = confusion_stats(y_true, y_pred)

        # 計算重要的指標
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (precision + recall) != 0 else 0

        # 印出
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
                y_score = y_score[:, 1] # 取出預測為 1 的機率
        except:
            try:
                y_score = model.predict_score(X_val)  #  for LinearRegression
            except:
                y_score = model.predict(X_val)   #  for Linear Regression

        # 準備去跑所有不同的threshold來找到最好的
        t_min, t_max = np.min(y_score), np.max(y_score)
        thresholds = np.linspace(t_min, t_max, 1001)


        # 開始跑for loop去找到最好的threshold，找的方法是Fbeta(這篇報告基本上適用beta=2)
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
        print("\nConfusion Matrix:")
        print(f"[[TN={TN:5d}, FP={FP:5d}]\n [FN={FN:5d}, TP={TP:5d}]]")
        print("\nEvaluation Metrics:")
        print(f"  Accuracy:      {(TP + TN) / (TP + TN + FP + FN):.4f}")
        print(f"  Precision:     {precision:.4f}")
        print(f"  Recall:        {recall:.4f}")
        print(f"  F1 Score:      {f1:.4f}")
        print(f"  F{beta:.1f} Score:   {best_fbeta:.4f}")
        print("=" * 40 + "\n")

        return best_threshold

# 把confusion matrix印出來好看的模式
def plot_confusion_matrix(TP, FP, TN, FN, labels=["Negative", "Positive"]):
    cm = np.array([[TN, FP],
                   [FN, TP]])
    
    group_names = [["TN", "FP"],
                   ["FN", "TP"]]

    # 加上數字和標籤
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


# Logistic Regression
# ----------------------------------
class LogisticRegression:
    def __init__(self, lr=0.01, epoch=10000, verbose=False):
        self.lr = lr
        self.epoch = epoch
        self.verbose = verbose

    # sigmoid 將輸出轉換成機率值
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 進行訓練的主函式
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        np.random.seed(42)
        self.b = 0.0
        self.w = np.random.randn(X_train.shape[1])

        # 儲存歷史參數與 loss
        self.b_history = [self.b]
        self.w_history = [self.w.copy()]
        self.loss_history = []
        self.val_loss_history = []

        for e in range(self.epoch):
            # 前向預測
            z = X_train @ self.w + self.b
            p = self.sigmoid(z)
            error = p - y_train

            # 計算梯度
            b_grad = np.mean(error)
            w_grad = X_train.T @ error / len(X_train)

            # 更新參數
            self.b -= self.lr * b_grad
            self.w -= self.lr * w_grad

            # 計算 training loss
            # 我們用的loss function是Binary Cross-Entropy
            loss = -np.mean(y_train * np.log(p + 1e-8) + (1 - y_train) * np.log(1 - p + 1e-8))
            self.loss_history.append(loss)
            self.b_history.append(self.b)
            self.w_history.append(self.w.copy())

            # 計算 validation loss 並做 early stopping 判斷
            if X_val is not None and y_val is not None:
                p_val = self.sigmoid(X_val @ self.w + self.b)
                val_loss = -np.mean(y_val * np.log(p_val + 1e-8) + (1 - y_val) * np.log(1 - p_val + 1e-8))
                self.val_loss_history.append(val_loss)

                # 邏輯是如果最近 100 個 epoch 的 val loss 變化太小就提前停止訓練
                if len(self.val_loss_history) >= 100:
                    recent = self.val_loss_history[-100:]
                    if max(recent) - min(recent) < 0.0001:
                        print(f"Early stopping at epoch {e} due to flat val loss")
                        break

            # 顯示訓練狀況
            if self.verbose and e % 100 == 0:
                print(f"Epoch {e} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | b: {self.b:.4f}")

        # 找出最佳參數
        best_epoch = np.argmin(self.val_loss_history if X_val is not None else self.loss_history)
        self.best_b = self.b_history[best_epoch]
        self.best_w = self.w_history[best_epoch]
        self.best_epoch = best_epoch

    # 回傳預測機率
    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.best_w) + self.best_b)

    # 根據閾值把機率轉成01的分類
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # 畫訓練跑epoch的loss曲線    
    def plot_loss(self):
        plt.plot(self.loss_history, label='Train Loss', color='red')
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validation Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


# Naive Bayes (Hybrid: Numeric + Binary)
# ----------------------------------
class NaiveBayes:
    def __init__(self, x, y, binary_cols=None):
        self.x = x
        self.y = y

        # 需要分開 binary 和 numeric 特徵，Binary因為已經是01資料所以很好分
        if binary_cols is None:
            self.binary_cols = [
                i for i in range(x.shape[1])
                if np.all(np.isin(np.unique(x[:, i]), [0, 1]))
            ]
        else:
            self.binary_cols = binary_cols

        self.numeric_cols = [
            i for i in range(x.shape[1]) if i not in self.binary_cols
        ]

    def fit(self):
        self.class_dict = {}
        self.class_prob = {}
        self.summaries = {}

        total_samples = len(self.y)

        for c in np.unique(self.y):
            # 區分不同類別的x
            x_c = self.x[self.y == c]
            self.class_dict[c] = x_c
            self.class_prob[c] = len(x_c) / total_samples

            # 每類別的 mean & std
            m = np.mean(x_c, axis=0)
            s = np.std(x_c, axis=0) + 1e-8  
            self.summaries[c] = np.vstack([m, s])

    # 計算每個樣本在每個類別下的 log likelihood（pdf為Gaussian或Bernoulli）
    def joint_log_likelihood(self, test):
        if test.ndim == 1:
            test = test.reshape(1, -1)

        log_likelihood = []

        for c in np.unique(self.y):
            # Prior
            log_class_prob = np.log(self.class_prob[c])
            mean = self.summaries[c][0]
            std = self.summaries[c][1]

            total_log = log_class_prob

            # Gaussian likelihood（for numeric)
            if self.numeric_cols:
                t_num = test[:, self.numeric_cols]
                m_num = mean[self.numeric_cols]
                s_num = std[self.numeric_cols]

                # Gaussian log likelihood
                gaussian_log = (
                    -0.5 * np.log(2 * pi * np.square(s_num)) -
                    0.5 * ((t_num - m_num) ** 2) / np.square(s_num)
                )

                # discrimant function
                total_log += np.sum(gaussian_log, axis=1)

            # Bernoulli likelihood（for binary)
            if self.binary_cols:
                t_bin = test[:, self.binary_cols]
                p_bin = mean[self.binary_cols]  # 平均值可視為 P(x=1|y)
                p_bin = np.clip(p_bin, 1e-6, 1 - 1e-6)  # 避免 log(0)，雖然應該不會出現

                # Bernoulli log likelihood
                bernoulli_log = (
                    t_bin * np.log(p_bin) + (1 - t_bin) * np.log(1 - p_bin)
                )

                # discrimant function
                total_log += np.sum(bernoulli_log, axis=1)

            log_likelihood.append(total_log)

        return np.array(log_likelihood).T  


    def predict_proba(self, test):
        # Posterior
        log_likelihood = self.joint_log_likelihood(test)
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood)
        probs = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        return probs

    def predict(self, test, threshold=0.5):
        probs = self.predict_proba(test)
        return (probs[:, 1] > threshold).astype(int)

#%%
# 找出在不同 threshold 下 F1 或 F2 最佳的點，並繪圖顯示
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

    # 計算所有 threshold 之下的 F1 或 F2 score
    def compute(self):
        f1s = []
        f2s = []

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

    # 畫出 Threshold vs F1 / F2 的變化圖
    def plot(self):
        if self.f1_scores == [] or self.f2_scores == []:
            self.compute()

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, self.f1_scores, label="F1 Score", color='orange')
        plt.plot(self.thresholds, self.f2_scores, label="F2 Score", color='blue')

        plt.plot(self.best_threshold_f1, self.max_f1, 'o', color='orange')
        plt.plot(self.best_threshold_f2, self.max_f2, 'o', color='blue')

        plt.axvline(self.best_threshold_f1, color='orange', linestyle='--', alpha=0.5,
                    label=f'Best F1 @ {self.best_threshold_f1:.3f}')
        plt.axvline(self.best_threshold_f2, color='blue', linestyle='--', alpha=0.5,
                    label=f'Best F2 @ {self.best_threshold_f2:.3f}')

        plt.text(self.best_threshold_f1 + 0.015, self.max_f1, f'{self.max_f1:.3f}', color='orange',
                 ha='left', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        plt.text(self.best_threshold_f2 - 0.015, self.max_f2, f'{self.max_f2:.3f}', color='blue',
                 ha='right', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold vs F1, F2 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 回傳最佳 threshold 與對應的分數
    def get_best_thresholds(self):
        if self.best_threshold_f1 is None or self.best_threshold_f2 is None:
            self.compute()
        return {
            "best_threshold_f1": self.best_threshold_f1,
            "max_f1": self.max_f1,
            "best_threshold_f2": self.best_threshold_f2,
            "max_f2": self.max_f2
        }