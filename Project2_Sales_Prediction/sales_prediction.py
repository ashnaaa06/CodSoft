import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib 

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv("advertising.csv")  

# -------------------- Data Overview --------------------
print("\n🔹 First 5 Rows of Data:")
print(df.head())

print("\n🔹 Data Description:")
print(df.describe())

# -------------------- Visualize Correlations --------------------
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------- Train-Test Split --------------------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Train Random Forest Model --------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
for name, m in models.items():
    m.fit(X_train, y_train)
    y_p = m.predict(X_test)
    print(f"{name} R² Score: {r2_score(y_test, y_p):.4f}")

# -------------------- Accuracy Metrics --------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\n Model Accuracy:")
print(f"R² Score  : {r2:.4f}")
print(f"MSE       : {mse:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")

# -------------------- Feature Importance --------------------
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance (Random Forest)")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, color='purple')
plt.title("Residual Distribution (Actual - Predicted)")
plt.xlabel("Residuals")
plt.grid(True)
plt.show()

for feature in X.columns:
    plt.figure()
    sns.scatterplot(data=df, x=feature, y='Sales')
    plt.title(f"Sales vs {feature}")
    plt.grid(True)
    plt.show()


# -------------------- Actual vs Predicted Plot --------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Random Forest: Actual vs Predicted Sales')
plt.legend()
plt.grid(True)
plt.show()


joblib.dump(model, "model.pkl")
print(" Model saved as model.pkl")
