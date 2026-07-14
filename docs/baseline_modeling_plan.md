# Baseline Modeling Plan: V1 Environment-Only HPAI Outbreak Risk Prediction

This document outlines the strategy for training and evaluating baseline (non-graph) machine learning models using the train-ready spatiotemporal modeling table.

## 1. Context and Class Imbalance Warning

The dataset exhibits extreme class imbalance:
* **Total Rows:** 237,538
* **Positive Cases (Outbreaks next month):** 381 (**0.1604%** of the dataset)
* **Negative Cases (No outbreaks next month):** 237,157 (**99.8396%** of the dataset)

> [!IMPORTANT]
> **Do not use standard classification Accuracy** as the primary performance metric. A naive model predicting all zeros achieves 99.84% accuracy but is completely useless.

---

## 2. Evaluation Split Strategy

To simulate a real-world forecasting environment and prevent temporal leakage, we use a **temporal split** rather than a random k-fold split:
* **Training Set:** Years **2022, 2023, and 2024** (months 1-12 of each year).
* **Testing Set:** Year **2025** (months 1-11; December 2025 was dropped due to missing target values).

This ensures the models are tested on entirely unseen future periods, validating their predictive capabilities over time.

---

## 3. Evaluation Metrics

Models will be compared using the following imbalance-aware metrics:
1. **Average Precision (AP) / Area Under the Precision-Recall Curve (PR-AUC):** Summarizes precision-recall trade-offs (preferred metric for highly skewed targets).
2. **ROC-AUC:** Measures the model's ability to rank risk, although it can be overly optimistic under severe imbalance.
3. **Top-K Recall & Precision:** Evaluates performance at the highest-risk thresholds:
   - What fraction of actual outbreaks are captured in the top K highest-risk predicted CSD-months? (Recall)
   - What fraction of the top K highest-risk predictions are actual outbreaks? (Precision)
   - Useful values to evaluate: K values: 50, 100, 200.
4. **Confusion Matrix:** Evaluates true positives, false positives, true negatives, and false negatives at a selected probability threshold (e.g. optimized for recall).

---

## 4. Modeling Algorithms

We will train three baseline models:

### 1. Simple Time-Aware Baseline
* **Logic:** Predict risk for CSD-month M in year Y using the historical outbreak status of the same CSD in the same month of the previous year Y-1.
* **Purpose:** Establish the performance of a naive seasonal persistence model.

### 2. Logistic Regression (Weighted)
* **Logic:** Standard linear classifier with L2 regularization.
* **Imbalance handling:** Set `class_weight='balanced'` to scale loss inversely proportional to class frequencies.
* **Purpose:** Fast linear baseline.

### 3. Tree Ensemble (Random Forest / Gradient Boosting)
* **Logic:** Use `scikit-learn`'s `HistGradientBoostingClassifier` or `RandomForestClassifier`.
* **Imbalance handling:** Use class_weight when supported (e.g. RandomForestClassifier), or sample weights when using HistGradientBoostingClassifier.
* **Purpose:** Capture non-linear feature interactions (e.g. climate-land cover combinations).

---

## 5. Feature Engineering and Leakage Rules

* **Metadata Columns (Excluded from Training Features):** `csd_uid`, `csd_name`, `province_code`, `province_name`, `year`, `month`.
* **Feature Processing:** Standardize features using a `StandardScaler` (fit on train split only to avoid data leakage to the test split).
* **Leakage Checks:** Verify that no future labels or current-month outbreak labels are accidentally introduced as features.
