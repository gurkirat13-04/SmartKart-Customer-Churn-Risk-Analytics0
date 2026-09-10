# SmartKart — Customer Churn Risk Prediction & Retention Analytics

An end-to-end machine learning pipeline that identifies customers at risk of churning and converts model predictions into an actionable customer retention risk report.

## Overview

Customer churn is a major business challenge for retail companies. SmartKart uses supervised machine learning to predict whether a customer is likely to churn based on key behavioral and demographic factors.

The project demonstrates a complete ML workflow, starting with intentionally messy customer data and ending with a ranked churn-risk report that can support retention decisions.

## Business Objective

The goal is to help SmartKart's retention team:

* Identify customers likely to churn
* Prioritize high-risk customers for intervention
* Understand factors associated with churn
* Convert ML predictions into actionable business insights

## Dataset

The dataset contains **100 customer records** with intentionally messy data, including:

* Duplicate records
* Missing values
* Inconsistent data types
* Invalid values
* Extreme outliers

### Features

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| Customer_ID   | Unique customer identifier               |
| Age           | Customer age                             |
| Monthly_Spend | Customer's monthly spending              |
| Complaints    | Number of customer complaints            |
| Churn         | Target variable: 1 = Churn, 0 = No Churn |

## Machine Learning Pipeline

The project follows a **15-step ML pipeline**:

1. Data Collection
2. Data Understanding & Inspection
3. Data Cleaning
4. Outlier Detection & Treatment
5. Feature Selection
6. Target Definition
7. Target Encoding Verification
8. Train-Test Split
9. Feature Standardisation
10. Model Building
11. Model Training
12. Prediction
13. Model Evaluation
14. Model Interpretation
15. Business-Ready Final Output

## Data Preprocessing

The cleaning process includes:

* Removing duplicate records
* Stripping unnecessary whitespace
* Converting Age to numeric format
* Correcting `"thirty"` to `30`
* Handling impossible age values
* Replacing negative spending values
* Median imputation for missing values
* IQR-based outlier detection
* Outlier capping instead of deleting records

After cleaning, the dataset contains **95 records**.

## Model

### Logistic Regression

Logistic Regression was selected because churn is a **binary classification problem** and the model provides interpretable coefficients and churn probabilities.

### Selected Features

* Age
* Monthly_Spend
* Complaints

`Customer_ID` was excluded because it is an identifier rather than a predictive feature.

The dataset is divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to preserve the churn distribution.

Features are standardised using `StandardScaler`, fitted only on the training data to avoid data leakage.

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The notebook reports approximately **89–95% accuracy** and around **100% recall** on the test set.

> **Note:** The test set contains only 19 customers, so these performance metrics should be treated as indicative rather than representative of production-level performance.

## Key Business Insights

The model's coefficients provide useful directional insights:

* **Monthly Spend:** Higher spending is associated with lower churn risk.
* **Complaints:** More complaints are associated with higher churn risk.
* **Age:** Shows a comparatively small positive association with churn risk in this dataset.

### Retention Takeaway

SmartKart should prioritize **complaint resolution** and protecting relationships with **high-value customers** as important retention levers.

## Business-Ready Output

The final pipeline generates:

`smartkart_churn_risk_report.csv`

The report contains:

* Customer ID
* Age
* Monthly Spend
* Complaints
* Actual Churn
* Predicted Churn
* Churn Probability
* Risk Label

Customers are ranked by **Churn Probability**, allowing the retention team to focus first on the highest-risk customers.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab
* Jupyter Notebook

## Project Structure

```text
smartkart-churn-risk-ml-pipeline/
│
├── SmartKart_Churn_Prediction_ML_Pipeline.ipynb
├── SmartKart_dirty_100_rows.csv
├── smartkart_churn_risk_report.csv
└── README.md
```

## Future Improvements

For a larger real-world dataset, the project could be extended with:

* Additional behavioral and transactional features
* Cross-validation
* Hyperparameter tuning
* Class imbalance analysis
* Comparison with Random Forest, XGBoost and other classifiers
* ROC-AUC and PR-AUC analysis
* Model explainability using SHAP
* Automated retention recommendations
* Deployment as an API or dashboard

## Conclusion

SmartKart demonstrates how machine learning can transform imperfect customer data into a structured churn-risk analysis. The project combines data preprocessing, supervised learning, model evaluation and business interpretation to create an actionable retention workflow.
