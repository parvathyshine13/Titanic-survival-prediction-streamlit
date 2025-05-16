#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Loading the trained model
model = joblib.load('titanic_log_reg_model.pkl')

st.title("Titanic Survival Prediction")

# Collect user input features
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Convert user inputs into model features
# You need to preprocess like you did in training (encode categorical, etc.)

# Example for Sex encoding
sex_male = 1 if sex == 'male' else 0

# Example for Embarked one-hot encoding
embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Create input array for prediction - arrange features exactly as your model expects
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])

# Predict button
if st.button('Predict Survival'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f'The passenger is predicted to survive with probability {prediction_proba:.2f}')
    else:
        st.error(f'The passenger is predicted NOT to survive with probability {1 - prediction_proba:.2f}')


# In[ ]:





# In[ ]:




