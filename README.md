📊 Customer Churn Prediction

A machine learning project focused on analyzing customer behavior and predicting customer churn using the Telco Customer Churn dataset.

The project covers Exploratory Data Analysis (EDA), data visualization, feature analysis, and machine learning models to understand the factors that influence customer churn.

📌 Project Overview

Customer churn is an important problem for subscription-based businesses. Understanding why customers leave can help companies improve customer retention and make better business decisions.

In this project, customer data is analyzed to identify patterns and important factors associated with churn. Machine learning techniques are then used to build models capable of predicting whether a customer is likely to churn.

📂 Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Customers: 7,043

Features: 21

Target variable: Churn

Target classes: Yes / No

The dataset contains information about customer demographics, services, contracts, billing, and account details.

🔍 Exploratory Data Analysis

The first part of the project focuses on understanding the dataset and identifying patterns related to customer churn.

Key Findings

Approximately 26.54% of customers have churned.

Approximately 73.46% of customers have remained.

Customer tenure ranges from 0 to 72 months.

Average customer tenure is around 32 months.

Contract type shows noticeable differences in customer churn behavior.

Internet service type is associated with different churn patterns.

Payment methods were analyzed to compare churn behavior.

Monthly charges and total charges were explored in relation to churn.

Correlation analysis was performed on numerical features.

These findings help identify variables that may be useful for machine learning models.

🤖 Machine Learning

The second part of the project focuses on building and evaluating machine learning models for customer churn prediction.

The machine learning notebook is:

week2-ml-models.ipynb


The models are used to learn patterns from historical customer data and predict whether a customer is likely to churn.

📁 Project Structure
project-1/
│
├── README.md
├── notebookab4e09e9cf.ipynb
└── week2-ml-models.ipynb

Files

notebookab4e09e9cf.ipynb

Contains the exploratory data analysis, data inspection, visualizations, and initial findings.

week2-ml-models.ipynb

Contains the machine learning model development and analysis.

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook

Kaggle Dataset

⚙️ Installation

Clone the repository:

git clone https://github.com/kayanifahad954-afk/project-1.git


Navigate to the project directory:

cd project-1


Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter


Start Jupyter Notebook:

jupyter notebook


Then open either notebook:

notebookab4e09e9cf.ipynb


or

week2-ml-models.ipynb

📈 Project Goals

The main objectives of this project are:

Understand customer churn patterns.

Perform exploratory data analysis.

Identify important churn-related features.

Visualize relationships between customer attributes and churn.

Build machine learning models for churn prediction.

Evaluate model performance.

Gain practical experience with a real-world classification problem.

💡 Business Applications

A churn prediction system can help businesses:

Identify customers who may be at risk of leaving.

Understand factors contributing to customer churn.

Develop targeted customer-retention strategies.

Improve customer satisfaction.

Reduce potential revenue loss.

🚀 Future Improvements

Possible improvements for this project include:

Hyperparameter tuning.

Feature engineering.

Testing additional machine learning algorithms.

Handling class imbalance more extensively.

Comparing multiple models using consistent evaluation metrics.

Building an interactive churn prediction dashboard.

Deploying the trained model as a web application.

👨‍💻 Author

Kayan Ifahad

GitHub:
https://github.com/kayanifahad954-afk

📄 License

This project is created for educational and learning purposes.
