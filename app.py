import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
import pickle

with open("loan_classifier.pkl", "rb") as f:
    model = pickle.load(f)
scaler = joblib.load("std_scaler.bin")

def predict_loan_status(features):
    """
    Predicts loan status based on input features.
    """
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    return "Loan Fully Paid" if prediction == 1 else "Loan Not Fully Paid"

# Streamlit UI
st.title("Loan Prediction App")
st.write("Fill in the details below to check loan status prediction.")

# Input fields
int_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
installment = st.number_input("Installment Amount", min_value=0.0, value=500.0)
log_annual_inc = st.number_input("Log of Annual Income", min_value=0.0, value=10.0)
dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0)
fico = st.slider("FICO Score", 300, 850, 650)
revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=10000.0)
revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
inq_last_6mths = st.slider("Inquiries in Last 6 Months", 0, 10, 2)
delinq_2yrs = st.slider("Delinquencies in Last 2 Years", 0, 10, 0)
pub_rec = st.slider("Public Records", 0, 5, 0)
installment_to_income_ratio = st.number_input("Installment to Income Ratio", min_value=0.0, value=0.2)
credit_history = st.number_input("Credit History Length (years)", min_value=0.0, value=5.0)

# Predict button
if st.button("Predict Loan Status"):
    features = [
        int_rate, installment, log_annual_inc, dti, fico, revol_bal,
        revol_util, inq_last_6mths, delinq_2yrs, pub_rec,
        installment_to_income_ratio, credit_history
    ]
    result = predict_loan_status(features)
    st.success(f"Prediction: {result}")
