 Titanic Survival Prediction - Logistic Regression Model


**OVERVIEW**  
This project uses machine learning to predict passenger survival on the Titanic. The dataset includes demographic and travel-related information for each passenger. By cleaning the data and applying a logistic regression model, we aim to identify the key factors that influenced survival rates.



**PROBLEM STATEMENT**  
Predict whether a passenger survived the Titanic shipwreck based on features such as age, gender, ticket class, and family connections. This is a binary classification problem (Survived: 1 or 0).


**FEATURES USED**  
After data cleaning and feature engineering, the following features were used:  
- Pclass (Passenger class)  
- Sex (Encoded: male=1, female=0)  
- Age (Missing values filled with median)  
- SibSp (Siblings/Spouses aboard)  
- Parch (Parents/Children aboard)  
- Fare (Ticket fare)  
- Embarked (Port: C=0, Q=1, S=2)  
- Title (Extracted from name and encoded)  
- FamilySize (SibSp + Parch + 1)  
- IsAlone (1 = Alone, 0 = Not Alone)



**DATA PREPROCESSING**  
- Removed irrelevant columns: Cabin, PassengerId, Name, Ticket  
- Filled missing Age with median, and Embarked with mode  
- Extracted and encoded title from names  
- Mapped categorical variables to numerical values  
- Scaled features using StandardScaler



**TOOLS & TECHNOLOGIES USED**  
**Language:** Python  
**Libraries:**  
pandas, numpy, matplotlib, seaborn, scikit-learn



**METHODOLOGY**  
1. Data Cleaning & Exploration  
2. Feature Engineering  
3. Train-Test Split (80-20)  
4. Feature Scaling  
5. Logistic Regression Model Training  
6. Model Evaluation (Accuracy, Classification Report, Confusion Matrix)  
7. Cross-Validation (cv=5)  
8. Visualization of Survival Rates



**VISUALIZATION**  
**Survival Count by Sex**  
A bar chart visualizing the number of male and female passengers who survived vs. didn’t survive.



**MODEL PERFORMANCE**  
**Accuracy Score:** 78.7% 

**Cross-Validation Score:** 81%



**CONCLUSION**  
This logistic regression model effectively predicts survival using engineered features. Key influencers of survival included gender, ticket class, and whether the passenger was traveling alone.



**ACKNOWLEDGEMENTS**  
This project was created as part of the **CodSoft Data Science Internship**.
