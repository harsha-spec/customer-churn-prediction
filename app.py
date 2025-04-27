import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.title("Customer Churn Prediction App")
st.subheader("Predict churn using customer subscription and activity data")

# Sidebar input
st.sidebar.header('Customer Information')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 70, 30)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    subscription_length = st.sidebar.slider('Subscription Length (months)', 1, 60, 12)
    subscription_type = st.sidebar.selectbox('Subscription Type', ('Basic', 'Standard', 'Premium'))
    number_of_logins = st.sidebar.slider('Number of Logins', 0, 500, 50)
    login_activity = st.sidebar.selectbox('Login Activity', ('Active', 'Inactive'))
    customer_ratings = st.sidebar.slider('Customer Ratings', 1.0, 5.0, 3.5)

    # Encoding
    gender = 1 if gender == 'Male' else 0
    subscription_type = {'Basic': 0, 'Standard': 1, 'Premium': 2}[subscription_type]
    login_activity = 0 if login_activity == 'Active' else 1

    data = {
        'age': age,
        'gender': gender,
        'subscription_length': subscription_length,
        'subscription_type': subscription_type,
        'number_of_logins': number_of_logins,
        'login_activity': login_activity,
        'customer_ratings': customer_ratings
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main panel
st.subheader('User Input Parameters')
st.write(input_df)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
churn_labels = np.array(['No', 'Yes'])
st.write(f"Prediction: **{churn_labels[prediction][0]}**")

st.subheader('Prediction Probability')
st.write(f"Churn: {prediction_proba[0][1]*100:.2f}%")
st.write(f"No Churn: {prediction_proba[0][0]*100:.2f}%")

# CSV Upload Prediction
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload your CSV file (with proper columns)", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(batch_data.head())

    # Encoding for batch
    batch_data['gender'] = batch_data['gender'].map({'Male': 1, 'Female': 0})
    batch_data['subscription_type'] = batch_data['subscription_type'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
    batch_data['login_activity'] = batch_data['login_activity'].map({'Active': 0, 'Inactive': 1})

    batch_scaled = scaler.transform(batch_data[['age', 'gender', 'subscription_length', 'subscription_type', 'number_of_logins', 'login_activity', 'customer_ratings']])
    batch_prediction = model.predict(batch_scaled)

    batch_data['churn_prediction'] = churn_labels[batch_prediction]
    st.write("Prediction Results:")
    st.write(batch_data)

    st.download_button("Download Predictions as CSV", batch_data.to_csv(index=False), "predictions.csv", "text/csv")
