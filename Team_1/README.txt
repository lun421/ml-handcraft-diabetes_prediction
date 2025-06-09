本專案使用美國 CDC BRFSS 2015 調查資料，建構預測糖尿病風險的模型，並比較 Least Square、Naive Bayes、Logistic Regression 與 Logistic Regression（含 Undersampling）四種模型在分類指標上的表現。

資料共包含 253,680 筆樣本與 22 個特徵欄位。

----------------------------------------------------------------------------------------------------

各檔案說明如下：

- `README.txt`：本說明文件

- `Report_1.pdf`：書面報告

- `Slide_1.pdf`：簡報

- `Program`：資料夾

----------------------------------------------------------------------------------------------------

📂 Program 資料夾內容：

【原始資料】

- `diabetes_binary_health_indicators_BRFSS2015.csv`：原始資料集


【共用模組】

- `Preprocess.py`：資料前處理模組，包含：
  - `StratifiedSplitter`：依 label 分層切分資料（train/val/test）
  - `PartialStandardizer`：針對數值型欄位進行 Z-score 標準化

- `Models.py`：模型與評估函式庫，包含：
  - 分類模型類別：`LogisticRegression`, `NaiveBayes`
  - 評估工具：`ConfusionEvaluator`, `confusion_stats`, `FScoreThresholdAnalyzer`
  - 圖形工具：畫出 Loss 曲線、F-score vs Threshold 變化圖、Confusion Matrix 熱力圖


【執行檔案】
- `LeastSquare.py`：執行 Least Squares 模型

- `NaiveBayes.py`：執行自製 Naive Bayes 模型

- `LogisticGD.py`：執行 Logistic Regression 模型（Gradient Descent），含 Undersampling 

----------------------------------------------------------------------------------------------------

請依下列對應檔案執行所需模型訓練與評估流程：

- 執行 Least Squares 模型 → `LeastSquare.py`
- 執行 Naive Bayes 模型 → `NaiveBayes.py`
- 執行 Logistic Regression 模型（含 GD 與 Early Stopping）→ `LogisticGD.py`