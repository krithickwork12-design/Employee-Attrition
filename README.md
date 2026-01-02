#  Employee Attrition Analysis & Prediction

##  Project Overview
Employee attrition is a major challenge for organizations, leading to increased recruitment costs, productivity loss, and team instability.  
This project performs **end-to-end analysis and prediction of employee attrition** using data science and machine learning techniques, and deploys the solution using a **Streamlit web application**.

---

##  Business Objective
- Analyze employee data to understand **why employees leave**
- Identify **key factors driving attrition**
- Build a **machine learning model** to predict attrition risk
- Provide **actionable insights** for HR retention strategies

---

##  Dataset
- **Dataset Name:** Employee Attrition Dataset  
- **Target Variable:** `Attrition` (Yes / No)

###  Key Features
- Demographics: Age, Gender, MaritalStatus  
- Job Details: Department, JobRole, JobLevel  
- Compensation: MonthlyIncome  
- Work Factors: OverTime, WorkLifeBalance  
- Experience: YearsAtCompany, TotalWorkingYears  

---

##  Tools & Technologies
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Streamlit**
- **Pickle**

---

##  Project Workflow

### 1️ Data Cleaning
- Removed duplicates
- Dropped irrelevant columns
- Verified missing values
- Corrected data types

### 2️ Exploratory Data Analysis (EDA)
- Univariate, Bivariate, and Multivariate analysis
- Attrition analysis by income, overtime, department, and experience
- Correlation heatmap for numerical features

### 3️ Feature Engineering
- Label Encoding for binary variables
- One-Hot Encoding for categorical variables
- Feature scaling using StandardScaler
- Handled class imbalance using SMOTE

### 4️ Model Building
Models trained and evaluated:
- Logistic Regression
- Random Forest Classifier

### 5️ Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Score
- Feature importance analysis

### 6️ Deployment
- Built and deployed an **interactive Streamlit app**
- Users can input employee details and get:
  - Attrition prediction
  - Probability score

---

##  Key Insights
- Employees working **overtime** show higher attrition
- **Lower income** employees are more likely to leave
- **Early-career employees** have higher attrition risk
- Job role and experience significantly influence attrition


```bash
streamlit run app/app.py
