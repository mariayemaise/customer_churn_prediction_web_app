# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:25:14 2024

@author: Ajose Maria
"""

import pandas as pd
import pickle
import gzip
import streamlit as st

# Load the model and features
with gzip.open('customer_churn_model.pkl.gz', 'rb') as f:
    model_data = pickle.load(f)
    loaded_model = model_data['model']
    features = model_data['features_names']

# Load label encoders
with open("label_encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Function for prediction
def customer_churn_prediction(input_data):
    input_data_df = pd.DataFrame([input_data], columns=features)
    
    # Encode categorical features
    for column, encoder in encoders.items():
        if column in input_data_df:
            input_data_df[column] = encoder.transform([input_data_df[column][0]])

    # Make prediction
    prediction = loaded_model.predict(input_data_df)

    return 'Churn' if prediction[0] == 1 else 'No Churn'

# Main function for Streamlit app
def main():
    st.title('Customer Churn Prediction Web App')

    # Collect user inputs
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    SeniorCitizen = st.number_input('SeniorCitizen (0 for No, 1 for Yes)', min_value=0, max_value=1, step=1)
    Partner = st.selectbox('Partner', ['Yes', 'No'])
    Dependents = st.selectbox('Dependents', ['Yes', 'No'])
    Tenure = st.number_input('Tenure (months)', min_value=0)
    PhoneService = st.selectbox('PhoneService', ['Yes', 'No'])
    MultipleLines = st.selectbox('MultipleLines', ['No phone service', 'No', 'Yes'])
    InternetService = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox('OnlineSecurity', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.selectbox('OnlineBackup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.selectbox('DeviceProtection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.selectbox('TechSupport', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.selectbox('StreamingTV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.selectbox('StreamingMovies', ['No', 'Yes', 'No internet service'])
    Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox('PaperlessBilling', ['Yes', 'No'])
    PaymentMethod = st.selectbox(
        'PaymentMethod',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    )
    MonthlyCharges = st.number_input('MonthlyCharges ($)', min_value=0.0)
    TotalCharges = st.number_input('TotalCharges ($)', min_value=0.0)

    # Prediction button
    if st.button('Customer Churn Prediction'):
        input_data = {
            'gender': Gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': Tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
        }

        prediction = customer_churn_prediction(input_data)
        st.success(f'The prediction is: {prediction}')

# Run the app
if __name__ == "__main__":
    main()
