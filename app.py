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
# -----------------------------
# User Inputs
# -----------------------------

age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 200000, 30000)
years_at_company = st.slider("Years at Company", 0, 40, 5)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
overtime = st.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
job_involvement = st.slider("Job Involvement (1–4)", 1, 4, 3)
environment_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance (1–4)", 1, 4, 3)
total_working_years = st.slider("Total Working Years", 0, 40, 10)
years_in_current_role = st.slider("Years In Current Role", 0, 20, 5)
years_with_current_manager = st.slider("Years With Current Manager", 0, 20, 5)
distance_from_home = st.slider("Distance From Home", 1, 50, 10)
num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)

# Encode Binary
overtime = 1 if overtime == "Yes" else 0

# Encode Marital Status (if training used OneHot)
marital_single = 1 if marital_status == "Single" else 0
marital_married = 1 if marital_status == "Married" else 0
marital_divorced = 1 if marital_status == "Divorced" else 0

input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobLevel": job_level,
    "StockOptionLevel": stock_option_level,
    "OverTime": overtime,
    "JobSatisfaction": job_satisfaction,
    "JobInvolvement": job_involvement,
    "EnvironmentSatisfaction": environment_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "TotalWorkingYears": total_working_years,
    "YearsInCurrentRole": years_in_current_role,
    "YearsWithCurrManager": years_with_current_manager,
    "DistanceFromHome": distance_from_home,
    "NumCompaniesWorked": num_companies_worked,
    "MaritalStatus_Single": marital_single,
    "MaritalStatus_Married": marital_married,
    "MaritalStatus_Divorced": marital_divorced
}

input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0 

for key, value in input_dict.items():
    if key in input_df.columns:
        input_df.at[0, key] = value

input_scaled = scaler.transform(input_df)

if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Attrition Risk ({probability:.2%})")