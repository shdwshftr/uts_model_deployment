import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('RF_model.pkl')
gender_encode = joblib.load('gender_encode.pkl')
oneHot_encode = joblib.load('onehot_encode.pkl')
scaler = joblib.load('scaler.pkl')

def main():
    st.title('Churn Model Deployment')

    credit_score = st.number_input("CreditScore", 0, 1000)
    geography = st.radio("location", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 100)
    tenure = st.number_input("Tenure", 0, 10)
    balance = st.number_input("Balance", 0, 250000)
    num_of_products = st.number_input("Number of Products", 0, 4)
    has_cr_card = st.radio("Has credit card or not (0 for no, 1 for yes)", [0, 1])
    is_active_member = st.radio("are you an active member? (0 for no, 1 for yes)", [0, 1])
    estimated_salary = st.number_input("estimated salary", 0,200000)
    
    data = {'CreditScore': float(credit_score), 'Geography': geography, 'Gender': gender,'Age': int(age),
            'Tenure': int(tenure), 'Balance': float(balance),
            'NumOfProducts': int(num_of_products), 'HasCrCard': int(has_cr_card),
            'IsActiveMember': int(is_active_member), 'EstimatedSalary': float(estimated_salary)}
    
    df = pd.DataFrame([list(data.values())], columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                                                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    df = df.replace(gender_encode)
    geo = df[['Geography']]
    location = pd.DataFrame(oneHot_encode.transform(geo).toarray(),columns = oneHot_encode.get_feature_names_out())
    df = pd.concat([df, location], axis = 1)
    df = df.drop(['Geography'], axis = 1)
    
    if st.button('Make Prediction'):
        features = df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
