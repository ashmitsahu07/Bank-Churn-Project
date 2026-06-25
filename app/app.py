import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and expected columns
model = joblib.load(r'C:\Users\Ashmit\OneDrive\Desktop\Bank_Churn_project\notebooks\churn_model.pkl')
scaler = joblib.load(r'C:\Users\Ashmit\OneDrive\Desktop\Bank_Churn_project\notebooks\scaler.pkl')
model_columns = joblib.load(r'C:\Users\Ashmit\OneDrive\Desktop\Bank_Churn_project\notebooks\model_columns.pkl')

st.title("Predictive Intelligence: Bank Customer Churn")
st.markdown("Identify at-risk customers instantly and optimize retention strategies.")

# Sidebar Controls
st.sidebar.header("Customer Input Data")
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 250000.0, 50000.0)
salary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
is_active = st.sidebar.selectbox("Is Active Member?", [1, 0])
has_crcard = st.sidebar.selectbox("Has Credit Card?", [1, 0])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

if st.button("Generate Risk Score"):
    # 1. Structure the input data
    input_dict = {
        'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_products,
        'HasCrCard': has_crcard, 'IsActiveMember': is_active,
        'EstimatedSalary': salary, 'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain': 1 if geography == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0
    }
    
    # 2. Recreate Feature Engineering
    input_dict['Balance_Salary_Ratio'] = balance / salary if salary > 0 else 0
    input_dict['Product_Density'] = num_products / (tenure + 0.1)
    input_dict['Engagement_Product_Interaction'] = is_active * num_products
    input_dict['Age_Tenure_Interaction'] = age * tenure
    
    # 3. Create DataFrame and align columns
    input_df = pd.DataFrame([input_dict])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns] # Reorder to match training
    
    # 4. Scale and Predict
    scaled_input = scaler.transform(input_df)
    churn_prob = model.predict_proba(scaled_input)[0][1] * 100
    
    # 5. Display Results
    st.subheader("Risk Analysis Results")
    st.metric(label="Probability of Churn", value=f"{churn_prob:.1f}%")
    
    if churn_prob > 50:
        st.error("High Risk: Immediate retention action recommended.")
    else:
        st.success("Low Risk: Customer relationship is stable.")