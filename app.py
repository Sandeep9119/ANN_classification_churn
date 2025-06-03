import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf 

# Load the pre-trained model
model = tf.keras.models.load_model('ann_model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('le.pkl', 'rb') as f:
    le = pickle.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
# Input fields for customer details

credit_score = st.number_input('CreditScore')
geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',le.classes_)
age = st.slider('Age',18,92)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
num_of_products = st.slider('NumOfProducts',1,4)
has_cr_card = st.selectbox('HasCrCard',[0,1])
estimated_salary = st.number_input('EstimatedSalary')
is_active_member = st.selectbox('Is Active Member', [0, 1])
# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    
})

# Encode categorical variables
geo_encoded = ohe.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f"Churn Probability: {prediction_proba:.2f} (Customer is likely to churn)")
else:
    st.write(f"Churn Probability: {prediction_proba:.2f} (Customer is unlikely to churn)")