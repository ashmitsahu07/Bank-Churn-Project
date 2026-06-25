🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn
📌 Project Overview

This project develops a Machine Learning-based customer churn prediction system for a retail bank. The model predicts the probability that a customer will leave the bank (churn) and provides insights into the key factors influencing customer retention.

The project combines data preprocessing, feature engineering, predictive modeling, explainability, and an interactive Streamlit web application to help banks identify high-risk customers and support proactive retention strategies.

🎯 Objectives
Predict customer churn with high accuracy.
Generate customer churn probability scores.
Identify important factors contributing to churn.
Compare multiple machine learning models.
Provide an interactive dashboard for real-time predictions.

📂 Dataset
The dataset contains customer information including:

Feature	Description
CreditScore	Customer credit score
Geography	Customer location
Gender	Male/Female
Age	Customer age
Tenure	Years with bank
Balance	Account balance
NumOfProducts	Number of bank products
HasCrCard	Credit card ownership
IsActiveMember	Customer activity status
EstimatedSalary	Annual salary
Exited	Target variable (1 = Churn, 0 = Retained)

⚙️ Project Workflow
1️⃣ Data Preprocessing
Removed duplicate records
Handled missing values
Dropped unnecessary columns
CustomerId
Surname
Year
One-Hot Encoding of categorical variables
Feature Scaling using StandardScaler
2️⃣ Feature Engineering

The following new features were created:

Balance-to-Salary Ratio
Product Density
Engagement-Product Interaction
Age-Tenure Interaction
3️⃣ Machine Learning Models

The following models were implemented:

Logistic Regression
Decision Tree
Random Forest
XGBoost (Final Selected Model)
4️⃣ Model Evaluation

Models were evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score
5️⃣ Explainability

Model explainability includes:

Feature Importance Analysis
SHAP Value Analysis
Partial Dependence Plot
🖥️ Streamlit Dashboard Features

The web application allows users to:

Input customer details
Predict churn probability
View customer risk category
Perform "What-if" analysis by modifying customer information
Display engineered feature values
Generate real-time churn predictions
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-Learn
XGBoost
SHAP
Matplotlib
Seaborn
Streamlit
Joblib
📁 Project Structure
Bank_Churn_Project
│
├── app
│   └── app.py
│
├── data
│   └── European_Bank.csv
│
├── models
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
├── notebooks
│   └── churn_analysis.ipynb
│
├── requirements.txt
│
└── README.md
🚀 Installation

Clone the repository:

git clone https://github.com/ashmitsahu07/Bank-Churn-Project.git

Move into the project directory:

cd Bank_Churn_Project

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/app1.py
📊 Model Output

The application provides:

Churn Probability
Customer Risk Category (Low / Moderate / High)
Personalized Retention Recommendations
📈 Future Improvements
Hyperparameter tuning
Deployment on cloud platforms
Automated model retraining
Real-time database integration
Advanced customer segmentation
👨‍💻 Author

Ashmit Sahu

B.Sc. Computer Science (Data Science)

RV University

⭐ Acknowledgements

This project was developed as part of an industry-oriented data science internship focused on predictive analytics and customer churn modeling.
