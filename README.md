# Loan Prediction App

This project is a **Loan Prediction App** that helps users predict whether a loan will be **fully paid** or **not fully paid** based on various financial factors. The application uses a trained machine learning model and is built using **Streamlit** for the front-end interface.Check out the live app here: [Loan Prediction App](https://loanpredict-sbfbu7qqxasv8bbcd9rjey.streamlit.app/)

## Features
- Predicts loan repayment status based on user inputs
- Uses **Streamlit** for an interactive and user-friendly interface
- Model trained with **scikit-learn** and deployed with **joblib**
- Includes feature scaling using **StandardScaler**

## Installation & Setup
To run the application locally, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/LoanPredict.git
cd LoanPredict
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
```bash
streamlit run app.py
```

## Model & Data Preprocessing
The loan prediction model is built using **Random Forest Classifier** and trained on a dataset with features such as:
- Interest Rate
- Installment Amount
- Log of Annual Income
- Debt-to-Income Ratio
- FICO Score
- Revolving Balance
- Credit History Length

The model was trained with **scikit-learn**, and feature scaling was applied using `StandardScaler`. The trained model and scaler are saved as:
- `loan_classifier.pkl` (the trained model)
- `std_scaler.bin` (the feature scaler)

## How It Works
1. Users enter financial details via the Streamlit interface.
2. The app scales the inputs using `std_scaler.bin`.
3. The model makes a prediction.
4. The result is displayed as **"Loan Fully Paid"** or **"Loan Not Fully Paid"**.


## Contribution
Feel free to contribute to improving this project!
- Fork the repository
- Create a feature branch
- Submit a pull request

## Contact
For any questions, reach out to me at [andreanagari36@gmail.com].

---
⭐ If you like this project, don't forget to **star** the repository!

