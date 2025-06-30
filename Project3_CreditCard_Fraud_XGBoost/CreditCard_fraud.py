import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
#  Load & Sample the Data
df = pd.read_csv("creditcard.csv")
df_sample = df.sample(n=20000, random_state=42)

#  Understand the Data
print(" Class distribution:")
print(df_sample['Class'].value_counts())


df_sample['Hour'] = (df_sample['Time'] // 3600) % 24
plt.figure(figsize=(10,5))
sns.boxplot(x='Hour', y='Amount', hue='Class', data=df_sample)
plt.title(" Fraud Amount Patterns by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Transaction Amount")
plt.legend(title='Class (0 = Normal, 1 = Fraud)')
plt.show()


# Visual: Fraud vs Non-Fraud bar chart
ax = sns.countplot(x='Class', data=df_sample)
ax.set_xticklabels(['Non-Fraud', 'Fraud'])
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,  
            int(bar.get_height()),
            ha='center', fontsize=10)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Count")
plt.show()


#  Feature-Target Split
X = df_sample.drop("Class", axis=1)
y = df_sample["Class"]
# Train-Test Split (70-30 with balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
# Train the Fraud Radar (XGBoost)
model = XGBClassifier(
    scale_pos_weight=99,  
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

#  Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# Outlier-based Feature Engineering
X_train['is_high_amount'] = (X_train['Amount'] > 500).astype(int)
X_train['is_night'] = ((X_train['Time'] // 3600 % 24 < 6) | (X_train['Time'] // 3600 % 24 > 22)).astype(int)

X_test['is_high_amount'] = (X_test['Amount'] > 500).astype(int)
X_test['is_night'] = ((X_test['Time'] // 3600 % 24 < 6) | (X_test['Time'] // 3600 % 24 > 22)).astype(int)


# Threshold tuning
custom_threshold = 0.35
y_pred_thresh = (y_proba > custom_threshold).astype(int)

# Outlier-based logic (manual rules)
outlier_flags = ((X_test['is_high_amount'] == 1) | (X_test['is_night'] == 1)).astype(int)

# Combine predictions: flag if either model or outlier rules say it's fraud
y_combined = ((y_pred_thresh == 1) | (outlier_flags == 1)).astype(int)

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

f1_scores = 2 * (precision * recall) / (precision + recall)

best_thresh = thresholds[np.argmax(f1_scores)]

y_pred_custom = (y_proba > best_thresh).astype(int)

#  Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(" Confusion Matrix:\n", conf_matrix)
print("\n Classification Report:\n", report)
print(f"\n AUC-ROC Score: {auc:.4f}")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")


#valuation After Combining Threshold + Outlier Logic
print(" Evaluation After Combining Threshold + Outlier Logic")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_combined))
print("\nClassification Report:\n", classification_report(y_test, y_combined))
print("Accuracy:", accuracy_score(y_test, y_combined))

# Visual: Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix: Fraud Radar")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Confusion matrix values
before_tuning = [5990, 0, 3, 7]      # [TN, FP, FN, TP]
after_tuning = [5134, 856, 1, 9]     # [TN, FP, FN, TP]

labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, before_tuning, width, label='Before Tuning', color='skyblue')
bar2 = ax.bar(x + width/2, after_tuning, width, label='After Tuning', color='salmon')

# Labels
ax.set_xlabel("Confusion Matrix Components")
ax.set_ylabel("Count")
ax.set_title(" Comparison of Confusion Matrix (Before vs After Tuning)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(bar1, padding=3)
ax.bar_label(bar2, padding=3)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# Visual: Feature Impact 
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title(" Feature Importance in Fraud Detection")
plt.xlabel("XGBoost Feature Impact")
plt.show()



