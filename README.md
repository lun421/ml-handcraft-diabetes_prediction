#  Handcrafted Machine Learning for Diabetes Prediction

A machine learning project built from scratch to predict diabetes risk using public health indicators, without relying on scikit-learn models. All models, training procedures, and evaluation metrics are manually implemented in Python.

## 📂 Project Structure


ml-handcraft-diabetes_prediction/
├── data/
│ └── diabetes_binary_health_indicators_BRFSS2015.csv
├── preprocess/
│ ├── preprocess.py # Data splitting and partial standardization
│ └── balance.py # Undersampling (1:1) function
├── models/
│ └── models.py # Logistic Regression (GD), Naive Bayes, Least Squares
├── evaluation/
│ ├── metrics.py # ConfusionEvaluator and confusion_stats
│ └── threshold.py # F1/F2 threshold optimizer and plotting
├── experiment/
│ ├── NaiveBayes.py # Experiment script for Naive Bayes
│ ├── LogisticGD.py # Experiment script for Logistic Regression
│ └── LeastSquare.py # Experiment script for OLS Linear Regression
├── .gitignore
├── README.md
└── requirements.txt
---

## 🧾 Dataset

- **Name:** [Behavioral Risk Factor Surveillance System (BRFSS) 2015](https://www.cdc.gov/brfss/index.html)
- **Source:** U.S. Centers for Disease Control and Prevention (CDC)
- **Rows:** 253,680 individuals  
- **Target Variable:** `Diabetes_binary` (1 if diagnosed with diabetes, 0 otherwise)  
- **Features:** Health indicators such as BMI, smoking status, exercise, general health, etc.

---

## 🧪 Models Implemented (from scratch)

### ✅ Logistic Regression (Gradient Descent)
- Binary cross-entropy loss
- Early stopping with validation loss monitoring
- Manual threshold tuning with F1/F2 optimization

### ✅ Naive Bayes
- Hybrid handling for both numeric (Gaussian) and binary (Bernoulli) features
- Posterior-based prediction and interpretability

### ✅ Linear Regression (Least Squares)
- Closed-form OLS solution
- Interpreted as a scoring model for classification (threshold optimization)

---

## 📊 Evaluation Approach

- Manual implementation of confusion matrix and evaluation metrics
- Threshold sweeping to find optimal F1 / F2 scores
- ROC/F-score plotting per model
- Emphasis on **Recall** and **F2 Score**, due to the public health context

---

## 🛠️ How to Run

### 1. Set working directory (in experiment scripts)
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Naive Bayes
python experiment/NaiveBayes.py

# Logistic Regression
python experiment/LogisticGD.py

# Least Squares Classification
python experiment/LeastSquare.py

📌 Key Highlights
No use of scikit-learn models — full low-level implementation

Modular design: clean separation of models, preprocess, and evaluation

Threshold selection designed around F2 score optimization

Educational value: great for those learning model internals

💻 Requirements
See requirements.txt, but generally:

Python ≥ 3.8

numpy

pandas

matplotlib

seaborn

🙏 Acknowledgements
Dataset from CDC: BRFSS 2015

This project was built for academic purposes to understand model internals and evaluate classifier performance on public health data.


📮 Contact
For questions or suggestions, feel free to open an issue or contact me via GitHub.