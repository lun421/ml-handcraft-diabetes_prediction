import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Logistic Regression
class LogisticRegression:
    """
    Logistic Regression classifier implemented using gradient descent.

    Features
    --------
    - Binary cross-entropy loss
    - Early stopping based on validation loss plateau
    - Tracks parameter and loss history

    Parameters
    ----------
    lr : float
        Learning rate (default=0.01)

    epoch : int
        Maximum number of training iterations (default=10000)

    verbose : bool
        Whether to print loss during training (default=False)

    Attributes
    ----------
    best_w : ndarray
        Best weight vector selected by validation loss

    best_b : float
        Best bias selected by validation loss

    loss_history : list of float
        Training loss at each epoch

    val_loss_history : list of float
        Validation loss at each epoch (if provided)

    Methods
    -------
    fit(X_train, y_train, X_val=None, y_val=None)
        Train the model

    predict_proba(X)
        Return predicted probabilities

    predict(X, threshold)
        Return binary predictions based on threshold

    plot_loss()
        Plot training/validation loss curve
    """
    def __init__(self, lr=0.01, epoch=10000, verbose=False):
        self.lr = lr
        self.epoch = epoch
        self.verbose = verbose

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        np.random.seed(42)
        self.b = 0.0
        self.w = np.random.randn(X_train.shape[1])

        self.b_history = [self.b]
        self.w_history = [self.w.copy()]
        self.loss_history = []
        self.val_loss_history = []

        for e in range(self.epoch):
            # Forward pass
            z = X_train @ self.w + self.b
            p = self.sigmoid(z)
            error = p - y_train

            # Compute gradients
            b_grad = np.mean(error)
            w_grad = X_train.T @ error / len(X_train)

            # Update parameters
            self.b -= self.lr * b_grad
            self.w -= self.lr * w_grad

            # Compute training loss w/  binary cross-entropy
            loss = -np.mean(y_train * np.log(p + 1e-8) + (1 - y_train) * np.log(1 - p + 1e-8))
            self.loss_history.append(loss)
            self.b_history.append(self.b)
            self.w_history.append(self.w.copy())

            # Compute validation loss
            if X_val is not None and y_val is not None:
                p_val = self.sigmoid(X_val @ self.w + self.b)
                val_loss = -np.mean(y_val * np.log(p_val + 1e-8) + (1 - y_val) * np.log(1 - p_val + 1e-8))
                self.val_loss_history.append(val_loss)

                # Early stopping: stop if validation loss flattens
                if len(self.val_loss_history) >= 100:
                    recent = self.val_loss_history[-100:]
                    if max(recent) - min(recent) < 0.0001:
                        print(f"Early stopping at epoch {e} due to flat val loss")
                        break

            if self.verbose and e % 100 == 0:
                print(f"Epoch {e} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | b: {self.b:.4f}")

        # Select best parameters based on minimum loss
        best_epoch = np.argmin(self.val_loss_history if X_val is not None else self.loss_history)
        self.best_b = self.b_history[best_epoch]
        self.best_w = self.w_history[best_epoch]
        self.best_epoch = best_epoch


    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.best_w) + self.best_b)

    def predict(self, X, threshold):
        return (self.predict_proba(X) >= threshold).astype(int)

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
class NaiveBayes:
    """
    Naive Bayes classifier for datasets with mixed binary and numeric features.

    Features
    --------
    - Numeric features are modeled with Gaussian distributions
    - Binary features are modeled with Bernoulli distributions

    Parameters
    ----------
    x : ndarray
        Feature matrix (n_samples, n_features)

    y : ndarray
        Target vector (n_samples,)

    binary_cols : list of int, optional
        Indices of binary features. If None, inferred from 0/1 check.

    Methods
    -------
    fit()
        Estimate parameters for each class

    predict_proba(X)
        Return class probabilities

    predict(X, threshold)
        Return binary predictions
    """
    def __init__(self, x, y, binary_cols=None):
        self.x = x
        self.y = y

        # Auto-detect binary columns
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
            x_c = self.x[self.y == c]
            self.class_dict[c] = x_c
            self.class_prob[c] = len(x_c) / total_samples

            m = np.mean(x_c, axis=0)
            s = np.std(x_c, axis=0) + 1e-8  
            self.summaries[c] = np.vstack([m, s])

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
                p_bin = mean[self.binary_cols]  
                p_bin = np.clip(p_bin, 1e-6, 1 - 1e-6)  

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

    def predict(self, test, threshold):
        probs = self.predict_proba(test)
        return (probs[:, 1] > threshold).astype(int)

# Least Squares Classifier
class LeastSquaresClassifier:
    """
    Linear classification using ordinary least squares (OLS).

    Features
    --------
    - Computes regression coefficients via normal equations
    - Converts regression output to binary prediction via threshold

    Methods
    -------
    fit(X, y)
        Solve for theta via OLS

    predict_score(X)
        Return raw regression scores

    predict(X, threshold)
        Return binary predictions
    """

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        self.theta = np.linalg.inv(XtX) @ Xty

    def predict_score(self, X):
        return np.dot(X, self.theta)

    def predict(self, X, threshold):
        return (self.predict_score(X) >= threshold).astype(int)