#  Credit Card Fraud Detection using XGBoost 

A machine learning project that leverages the powerful **XGBoost** algorithm to detect fraudulent credit card transactions. This project goes beyond standard modeling by incorporating **threshold tuning** and **outlier detection** to simulate real-world fraud detection systems.

---

## 🚀 Project Highlights

- ✅ Built and evaluated a classification model using **XGBoost**
- 🧠 Enhanced performance using **custom threshold tuning**
- ⚠️ Identified hidden frauds via **outlier analysis**
- 📊 Visualized **class imbalance**, **confusion matrix**, and **feature importance**
- 📈 Achieved **AUC-ROC Score of 0.9760** after combining tuning techniques
- 🧪 Used probabilistic output to fine-tune fraud recall vs precision trade-off

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Frauds**: 492 (≈ 0.172%)
- **Features**:
  - PCA-transformed: `V1` to `V28`
  - Time-related: `Time`, `Amount`
  - Label: `Class` (0 = Non-Fraud, 1 = Fraud)

---

## 🧪 Technologies Used

| Type         | Tools |
|--------------|-------|
| **Language** | Python |
| **Libraries** | `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn` |

---

## ⚙️ Model Overview

- **Model**: `XGBClassifier`
- **Why XGBoost?**
  - Handles class imbalance well
  - Fast and scalable
  - Outputs class probabilities for ROC & threshold tuning

---

## 🧠 Key Enhancements

### ✅ 1. **Custom Threshold Tuning**
- Instead of relying on the default `0.5` decision threshold, we tuned it to increase **recall**.
- Helps detect more **fraud cases** that may otherwise be missed.

### ✅ 2. **Outlier Detection**
- Analyzed transactions with abnormally high `Amount` or `Time` patterns.
- Integrated these conditions with model predictions to **boost fraud detection**.

### ✅ 3. **Combined Logic: XGBoost + Rules**
- Final predictions were improved by **combining model output** with **rule-based detection**.
- Result: More True Positives, fewer missed frauds.

---

## 📊 Visualizations Included

### 1. **Fraud vs Non-Fraud Transactions**
> Highlights dataset imbalance. Fraud transactions are rare — justifying use of AUC-ROC and recall over accuracy.

### 2. **Confusion Matrix Heatmap**
> Visual breakdown of classification performance. Helps compare true/false positives and negatives. Tuned threshold reduced False Negatives.

### 3. **Feature Importance**
> Displays which features XGBoost relied on most — such as `V14`, `V17`, and `V10`. Useful for explainability.

### 4. **Transaction Amount by Hour + Class**
> Outlier plots showed frauds tend to occur at off-peak hours with unusually high amounts.


---

## ✅ Results After Enhancements

| Metric         | Value   |
|----------------|---------|
| AUC-ROC Score  | 0.9760  |
| Accuracy       | 0.9870+ |
| False Negatives| Reduced from 3 to 1 |
| True Positives | Increased from 7 to 9 |

> 🎯 Combining threshold tuning + outlier logic successfully captured **additional hidden frauds**.

---

## 📦 Future Enhancements

- 🧪 Add **SHAP values** for better interpretability
- 🛠 Integrate **live stream detection**
- 🌐 Deploy using **Streamlit or Flask**
- 🔔 Add **SMS/Email alerts** using services like Twilio
- 🔁 Integrate **real-time retraining pipeline**


