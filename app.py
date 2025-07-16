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

import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("rf_model.pkl")

st.title("Insurance Claim Prediction")

# --- Single prediction (existing) ---
st.subheader("🔹 Predict for a Single Client")
# (keep your existing single-input form here...)

st.write("---")

# --- Batch prediction ---
st.subheader("🔹 Batch Prediction for Multiple Clients")

uploaded_file = st.file_uploader("Upload a CSV file with client data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    batch_data = pd.read_csv(uploaded_file)
    
    st.write("✅ Uploaded data preview:")
    st.write(batch_data.head())
    
    # Ensure the columns match the model
    # (Adjust this list to your actual feature columns)
    expected_cols = ["age", "vehicle_age", "annual_premium", "num_policies", "vehicle_type"]
    
    # Check for missing columns
    missing_cols = [col for col in expected_cols if col not in batch_data.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        # Make predictions
        predictions = model.predict(batch_data[expected_cols])
        
        # Add predictions as new column
        batch_data["Claim_Prediction"] = predictions
        
        # Show results
        st.write("### 📊 Predictions")
        st.write(batch_data.head())
        
        # Allow download
        csv_output = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv_output, file_name="predictions.csv", mime="text/csv")
