# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("rf_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("🛡️ Insurance Claim Prediction App")
st.write("Enter policyholder info to predict if they're likely to claim.")

# Sidebar input form
st.sidebar.header("📋 Policyholder Details")

age = st.sidebar.slider("Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
policy_type = st.sidebar.selectbox("Policy Type", ['Comprehensive', 'Third Party', 'Third Party, Fire and Theft'])
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 5)
annual_premium = st.sidebar.number_input("Annual Premium (ZAR)", 1000, 20000, 8000)
claims_history = st.sidebar.slider("Previous Claims", 0, 10, 1)
region = st.sidebar.selectbox("Region", ['Urban', 'Suburban', 'Rural'])
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
number_of_dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1)

# Format into DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'policy_type': [policy_type],
    'vehicle_age': [vehicle_age],
    'annual_premium': [annual_premium],
    'claims_history': [claims_history],
    'region': [region],
    'credit_score': [credit_score],
    'number_of_dependents': [number_of_dependents]
})

# One-hot encoding
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Predict
if st.button("🔮 Predict Claim Likelihood"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ This policyholder is LIKELY to claim. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This policyholder is UNLIKELY to claim. (Probability: {probability:.2f})")
