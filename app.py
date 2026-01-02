import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load artifacts
model = pickle.load(open("model/rf_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_columns.pkl", "rb"))

st.title(" Employee Attrition Prediction ")

# User Inputs
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 200000, 30000)
years_at_company = st.slider("Years at Company", 0, 40, 5)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
overtime = st.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance (1–4)", 1, 4, 3)

# Encode binary
overtime = 1 if overtime == "Yes" else 0

# Create input dictionary (RAW FEATURES)
input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobLevel": job_level,
    "OverTime": overtime,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance
}

# Create dataframe with ALL training columns
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0  # initialize all as 0

# Fill known values
for key, value in input_dict.items():
    if key in input_df.columns:
        input_df.at[0, key] = value

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Attrition Risk ({probability:.2%})")
