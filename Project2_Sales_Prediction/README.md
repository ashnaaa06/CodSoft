**Sales Prediction using Random Forest Regression**

 **OVERVIEW :**  
This project focuses on predicting product sales based on advertising budget across different media: TV, Radio, and Newspaper. By exploring the Advertising dataset and applying machine learning models, the project aims to determine which channels are most effective in driving sales.

 **PROBLEM STATEMENT :**  
Given the budget allocation across various media, predict the corresponding sales figures. The objective is to build a regression model that can accurately estimate future sales based on advertising spend.

 **DATASET USED :**  
**File:** advertising.csv  
**Attributes:**
- **TV** – Budget spent on TV advertising
- **Radio** – Budget spent on Radio advertising
- **Newspaper** – Budget spent on Newspaper advertising
- **Sales** – Target variable (Product sales in thousands of units)

 **FEATURES USED FOR MODELING :**  
- TV  
- Radio  
- Newspaper  

 **DATA PREPROCESSING AND EXPLORATION :**
- Loaded the dataset using pandas  
- Visualized correlation using a heatmap  
- Created scatter plots of each feature vs. sales  
- Checked for outliers using residual distribution plots

 **TOOLS & TECHNOLOGIES USED :**

**Language:**  
Python

**Libraries:**  
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

 **METHODOLOGY :**

1. **Data Loading** – Read the advertising dataset using pandas  
2. **Data Visualization** – Correlation heatmap and scatter plots  
3. **Train-Test Split** – Split data into 80% training and 20% testing  
4. **Model Training** – Used:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor  
5. **Model Evaluation** – R² Score, MSE, RMSE, MAE  
6. **Feature Importance** – Visualized most influential features  
7. **Residual Analysis** – Checked prediction errors  
8. **Model Export** – Saved the trained model with joblib



 **VISUALIZATION OUTPUTS :**

- Correlation heatmap  
- Feature importance chart  
- Residual distribution  
- Actual vs Predicted Sales plot  
- Sales vs. individual media spend plots

 **CONCLUSION :**  
The Random Forest model performed best with high R² and low errors. TV and Radio budgets were the strongest predictors of sales, while Newspaper had minimal impact.

 **ACKNOWLEDGEMENT :**  
This project was completed as part of the **CodSoft Data Science Internship**.
