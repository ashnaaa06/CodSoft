import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic_data = pd.read_csv('C:/Users/ADMIN/OneDrive/Desktop/Titanic_Model/Titanic-dataset.csv')

titanic_data.drop(['Cabin','PassengerId'], axis=1, inplace=True)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

name_backup = titanic_data['Name'].copy()
titanic_data['Title'] = name_backup.str.extract(r' ([A-Za-z]+)\.', expand=False)
titanic_data['Title'] = titanic_data['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
titanic_data['Title'] = titanic_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')
titanic_data['Title'] = titanic_data['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
titanic_data['Title'].fillna(0, inplace=True)

titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data['IsAlone'] = 1
titanic_data.loc[titanic_data['FamilySize'] > 1, 'IsAlone'] = 0
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 1, 'female': 0})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
titanic_data = titanic_data.drop(['Name'], axis=1)
titanic_data = titanic_data.drop(['Ticket'], axis=1)


X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 8: Evaluate
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5)
print("📈 Cross-Validation Accuracy:", scores.mean())

# Optional: Visualize survival by Sex
sns.countplot(x='Sex', hue='Survived', data=titanic_data, palette='Set2')
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
