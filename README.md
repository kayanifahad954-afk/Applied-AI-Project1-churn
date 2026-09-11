# Customer Churn Prediction 
 
## Week 1: Exploratory Data Analysis 
 
### Dataset - Source: Telco Customer Churn (Kaggle) - Size: 7,043 customers, 21 features - Target: Predict customer churn (Yes/No) 
 
### Key Findings 

- The dataset contains **7,043 customers** and **21 features**.
- Overall, **26.54% of customers churned**, while **73.46% remained**.
- Customer tenure varies from **0 to 72 months**, with an average tenure of about **32 months**.
- Churn analysis shows that **customer tenure is an important factor** in understanding churn behavior.
- The analysis examines **MonthlyCharges** and **TotalCharges** to understand their relationship with customer churn.
- **Contract type** is an important churn-related factor, with different contract groups showing different churn behavior.
- **Internet service type** also shows differences in customer churn.
- **Payment method** was analyzed to compare churn rates among different payment options.
- A **correlation heatmap** was created to identify relationships between numerical features and churn.
- Overall, the analysis suggests that **tenure, charges, contract type, internet service, and payment method** are important areas to consider when studying customer churn.
 
### Setup 
Open the Kaggle notebook or run locally: 
pip install pandas numpy matplotlib seaborn 
