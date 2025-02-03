import streamlit as st
import joblib
import numpy as np

# Configure the page layout and title
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title("Loan Prediction App")
st.write("Fill in the details below to predict whether a loan will be fully paid or not.")

# Load the trained model and scaler
model = joblib.load("loan_classifier.pkl")
scaler = joblib.load("std_scaler.bin")

# Create a form for input data
with st.form(key="loan_form"):
    st.header("Applicant Financial Details")
    
    # First row: Interest Rate, Installment, Log Annual Income, Debt-to-Income Ratio
    col1, col2 = st.columns(2)
    with col1:
        int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1, help="Enter the interest rate as a percentage.")
        installment = st.number_input("Installment Amount", min_value=0.0, value=500.0, step=100.0, help="Monthly installment amount.")
        log_annual_inc = st.number_input("Log of Annual Income", min_value=0.0, value=10.0, step=0.1, help="Logarithm of the annual income.")
    with col2:
        dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=15.0, step=0.1, help="Enter the debt-to-income ratio.")
        fico = st.slider("FICO Score", min_value=300, max_value=850, value=650, step=1, help="Enter the FICO credit score.")
    
    st.markdown("---")
    
    st.header("Credit and Loan History Details")
    # Second row: Revolving Balance, Revolving Utilization, Inquiries, Delinquencies, Public Records, Installment to Income Ratio, Credit History
    col3, col4 = st.columns(2)
    with col3:
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=10000.0, step=1000.0, help="The current revolving balance on the account.")
        revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, help="Revolving utilization percentage.")
        inq_last_6mths = st.slider("Inquiries in Last 6 Months", min_value=0, max_value=10, value=2, step=1, help="Number of credit inquiries in the last 6 months.")
    with col4:
        delinq_2yrs = st.slider("Delinquencies in Last 2 Years", min_value=0, max_value=10, value=0, step=1, help="Number of delinquencies in the past 2 years.")
        pub_rec = st.slider("Public Records", min_value=0, max_value=5, value=0, step=1, help="Number of derogatory public records.")
        installment_to_income_ratio = st.number_input("Installment to Income Ratio", min_value=0.0, value=0.2, step=0.1, help="Ratio of installment to income.")
        credit_history = st.number_input("Credit History (Years)", min_value=0.0, value=5.0, step=0.1, help="Length of credit history in years.")
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict Loan Status")

# Process the input and display the prediction
if submit_button:
    # Arrange inputs in the correct order as expected by your model
    features = [
        int_rate, installment, log_annual_inc, dti, fico,
        revol_bal, revol_util, inq_last_6mths, delinq_2yrs,
        pub_rec, installment_to_income_ratio, credit_history
    ]
    input_array = np.array([features])
    scaled_array = scaler.transform(input_array)
    prediction = model.predict(scaled_array)[0]
    
    if prediction == 1:
        st.success("Prediction: Loan Fully Paid")
    else:
        st.error("Prediction: Loan Not Fully Paid")
