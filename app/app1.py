import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Must be the very first command)
st.set_page_config(page_title="Churn Predictor", page_icon="🏦", layout="wide")

# 2. Load Artifacts efficiently
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    scaler =joblib.load("models/scaler.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

# 3. Main Dashboard Header
st.title("🏦 Predictive Intelligence: Customer Churn")
st.markdown("""
Welcome to the interactive Churn Prediction model. Adjust the customer metrics in the sidebar to see 
their real-time probability of leaving the bank. 
""")
st.divider()

# 4. Sidebar for Inputs
st.sidebar.header("⚙️ Customer Parameters")
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
salary = st.sidebar.number_input("Estimated Salary ($)", 10000.0, 200000.0, 60000.0, step=1000.0)

st.sidebar.markdown("---")
is_active = st.sidebar.selectbox("Active Member?", ["Yes", "No"])
has_crcard = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Convert text selections back to binary for the model
is_active_bin = 1 if is_active == "Yes" else 0
has_crcard_bin = 1 if has_crcard == "Yes" else 0

# 5. Main Layout: Two Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Customer Summary")
    st.markdown(f"**Demographics:** {gender}, Age {age}, located in {geography}")
    st.markdown(f"**Financials:** ${balance:,.2f} Balance | ${salary:,.2f} Salary")
    st.markdown(f"**Engagement:** {num_products} Products | Active: {is_active} | Credit Score: {credit_score}")
    
    # Show technical features inside an expander so it doesn't clutter the main view
    with st.expander("View Engineered Features (Behind the scenes)"):
        st.code(f"""
        Balance/Salary Ratio: {balance / (salary + 0.01):.2f}
        Product Density: {num_products / (tenure + 0.1):.2f}
        Engagement Interaction: {is_active_bin * num_products}
        """)

with col2:
    st.subheader("🎯 Risk Analysis")
    
    # Button to trigger calculation
    if st.button("Generate Risk Score", use_container_width=True, type="primary"):
        # Structure input
        input_data = {
            'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': has_crcard_bin, 'IsActiveMember': is_active_bin,
            'EstimatedSalary': salary, 
            'Geography_Germany': 1 if geography == 'Germany' else 0,
            'Geography_Spain': 1 if geography == 'Spain' else 0,
            'Gender_Male': 1 if gender == 'Male' else 0,
            'Balance_Salary_Ratio': balance / (salary + 0.01),
            'Product_Density': num_products / (tenure + 0.1),
            'Engagement_Product_Interaction': is_active_bin * num_products,
            'Age_Tenure_Interaction': age * tenure
        }
        
        # Process and Predict
        input_df = pd.DataFrame([input_data])
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[model_columns]
        scaled_input = scaler.transform(input_df)
        churn_prob = model.predict_proba(scaled_input)[0][1] 
        
        # Display Results visually
        prob_percentage = churn_prob * 100
        
        st.metric("Probability of Churn", f"{prob_percentage:.1f}%")
        st.progress(float(churn_prob)) # Native visual progress bar
        
        # Dynamic Actionable Insights
        if churn_prob > 0.5:
            st.error("🚨 **High Risk of Churn**")
            st.markdown("**Recommendation:** Immediate intervention required. Consider offering targeted retention incentives, fee waivers, or a dedicated account manager review.")
        elif churn_prob > 0.3:
            st.warning("⚠️ **Moderate Risk**")
            st.markdown("**Recommendation:** Monitor activity closely. Send re-engagement campaigns focusing on unutilized bank products.")
        else:
            st.success("✅ **Low Risk**")
            st.markdown("**Recommendation:** Customer relationship is stable. Ideal candidate for upselling premium wealth management services.")