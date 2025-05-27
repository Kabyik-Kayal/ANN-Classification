import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load models and encoders with error handling
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('model.h5')
        
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        
        with open('one_hot_encoder_geo.pkl', 'rb') as f:
            one_hot_encoder_geo = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model, label_encoder_gender, one_hot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, label_encoder_gender, one_hot_encoder_geo, scaler = load_models()

if model is None:
    st.stop()

# App header
st.title('🏦 Customer Churn Prediction')
st.markdown("Predict the likelihood of customer churn based on customer features.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    geography = st.selectbox('🌍 Geography', one_hot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('📅 Age', 18, 92, 35)

with col2:
    st.subheader("Financial Information")
    credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=850, value=650)
    balance = st.number_input('💰 Balance', min_value=0.0, value=0.0, format="%.2f")
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, format="%.2f")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Account Details")
    tenure = st.slider('⏰ Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 2)

with col4:
    st.subheader("Account Status")
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

# Prediction button
if st.button('🔮 Predict Churn', type='primary'):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary],
        })

        # One-hot encode geography
        geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine data
        input_data = pd.concat([input_data, geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Display results
        st.subheader("Prediction Results")
        # Results interpretation
        if prediction_proba > 0.5:
            st.error(f'⚠️ High churn risk: {prediction_proba:.1%}')
            st.markdown("**Recommendation:** Consider retention strategies for this customer.")
        else:
            st.success(f'✅ Low churn risk: {prediction_proba:.1%}')
            st.markdown("**Status:** Customer is likely to stay.")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add feature explanations
with st.expander("ℹ️ Feature Explanations"):
    st.markdown("""
    - **Geography**: Customer's country/region
    - **Gender**: Customer's gender
    - **Age**: Customer's age in years
    - **Credit Score**: Customer's credit score (300-850)
    - **Balance**: Current account balance
    - **Tenure**: Years as customer
    - **Number of Products**: Banking products held
    - **Has Credit Card**: Whether customer has a credit card
    - **Is Active Member**: Whether customer is active
    - **Estimated Salary**: Annual salary estimate
    """)
