import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score

#  Load & Sample the Data
df = pd.read_csv("creditcard.csv")
df_sample = df.sample(n=20000, random_state=42)

#  Understand the Data
print(" Class distribution:")
print(df_sample['Class'].value_counts())

# Visual: Fraud vs Non-Fraud bar chart
# Basic bar plot
ax = sns.countplot(x='Class', data=df_sample)

# Rename x-axis labels
ax.set_xticklabels(['Non-Fraud', 'Fraud'])

# Add count labels on top of bars
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,  
            int(bar.get_height()),
            ha='center', fontsize=10)

# Labels and title
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

#  Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(" Confusion Matrix:\n", conf_matrix)
print("\n Classification Report:\n", report)
print(f"\n AUC-ROC Score: {auc:.4f}")

# Visual: Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix: Fraud Radar")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

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


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
