#  Credit Card Fraud Detection using XGBoost

A machine learning project that leverages the powerful XGBoost algorithm to detect fraudulent credit card transactions. Given the highly imbalanced nature of real-world fraud detection datasets, this project places emphasis on evaluation metrics like AUC-ROC, precision, and recall.

---

##  Project Highlights

- ✅ Built and evaluated a classification model using **XGBoost**
- 📊 Visualized class imbalance and confusion matrix
- 📈 Achieved **AUC-ROC Score of 0.9650**
- 🔍 Focused on performance metrics important for imbalanced data

---

##  Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Frauds**: 492 (≈ 0.172%)
- **Features**:
  - PCA-transformed: `V1` to `V28`
  - `Time`, `Amount`
  - `Class`: Target (0 = Non-Fraud, 1 = Fraud)



##  Technologies Used

- **Language**: Python
- **Libraries**:
  - Data: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `xgboost`, `scikit-learn`

---

 **Model Overview**
 XGBoost Classifier

- Fast, efficient, and highly accurate
- Robust against class imbalance
- Provides probabilistic outputs used for ROC analysis

---
**🧠 Future Enhancements**
Add threshold tuning to reduce false negatives

Use SMOTE or undersampling to improve fraud recall

Deploy via Streamlit or Flask for interactive use

Integrate alert system (e.g., email/SMS)


**📊 Visualizations**
1. Fraud vs Non-Fraud Transactions
Description: Shows the imbalance in the dataset. Most transactions are legitimate (non-fraud), while fraudulent transactions make up a very small portion. This highlights the need for careful model evaluation, as accuracy alone can be misleading
2. Confusion Matrix Heatmap
Description: Displays true positives, false positives, true negatives, and false negatives. Helps in understanding how well the model is performing, especially on fraud detection. High true negatives and low false negatives are desirable.
3. Feature Importance
Description: Displays the top features used by XGBoost to classify transactions as fraud or non-fraud. Helps in understanding which factors (e.g., `V14`, `V17`, `V10`, etc.) contribute most to the decision-making process. Feature importance is ranked based on the average gain provided by each feature to the model.


