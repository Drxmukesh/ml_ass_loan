import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

train = pd.read_csv("train.csv")
train = train.dropna()
train['TotalApplicantIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train = pd.get_dummies(train, columns=['Gender', 'Married', 'Education', 
                                       'Self_Employed', 'Property_Area', 'Loan_Status'], 
                                       drop_first=True)
train = train.rename(columns={'Loan_Status_Y': 'Loan_Approved'})

train['Credit_History'] = train['Credit_History'].astype(int)

x = train[['Gender_Male', 'Married_Yes', 'TotalApplicantIncome',
            'LoanAmount', 'Credit_History', 'Education_Not Graduate', 
            'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']]
y = train['Loan_Approved']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10, shuffle=True)
model = RandomForestClassifier()
model.fit(x_train, y_train)

@st.cache_data
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History, 
               Education, Self_Employed, Property_Area):
    Gender = 1 if Gender == 'Male' else 0
    Married = 1 if Married == 'Married' else 0
    Credit_History = 1 if Credit_History == 'Credit History' else 0
    Education = 1 if Education == 'Not Graduate' else 0
    Self_Employed = 1 if Self_Employed == 'Yes' else 0
    Property_Area_Semiurban = 1 if Property_Area == 'Semiurban' else 0
    Property_Area_Urban = 1 if Property_Area == 'Urban' else 0
    
    
    pred_inputs = model.predict([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History,
                                  Education, Self_Employed, Property_Area_Semiurban, Property_Area_Urban]])
    
    return 'Congratulations, you have been approved the loan' if pred_inputs[0] == 1 else 'I am sorry, you have been denied the loan'

def main():
    st.title("Loan Approval Prediction")
    st.markdown("""<div style="background-color:blue;padding:10px"> 
                <h2 style="color:white;text-align:center;">
                Loan Approval Prediction ML App</h2> </div>""", unsafe_allow_html=True)
    
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Married = st.selectbox('Marital Status', ('Unmarried', 'Married'))
    Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
    Property_Area = st.selectbox('Property Area', ('Rural', 'Semiurban', 'Urban'))
    TotalApplicantIncome = st.number_input("Total Applicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Credit_History = st.selectbox('Credit History', ('No Credit History', 'Credit History'))
    
    if st.button("Predict"):
        result = prediction(Gender, Married, TotalApplicantIncome, LoanAmount, 
                            Credit_History, Education, Self_Employed, Property_Area)
        st.success(f'Final Decision : {result}')

if __name__ == '__main__':
    main()
