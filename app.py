#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Risk Calculator")

st.write("Enter patient clinical details to estimate heart disease risk.")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0,1])
chest_pain = st.selectbox("Chest Pain Type (1-4)", [1,2,3,4])
bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol Level", 100, 600, 200)
fasting_sugar = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
rest_ecg = st.selectbox("Resting ECG Result (0-2)", [0,1,2])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
st_depression = st.slider("ST Depression", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of ST Segment (1-3)", [1,2,3])
num_major_vessels = st.selectbox("Number of Major Vessels", [0,1,2,3])
thallium = st.selectbox("Thallium Test Result (3,6,7)", [3,6,7])

if st.button("Predict Risk"):

    patient_data = np.array([[age,sex,chest_pain,bp,cholesterol,
                              fasting_sugar,rest_ecg,max_hr,
                              exercise_angina,st_depression,
                              st_slope,num_major_vessels,thallium]])

    patient_scaled = scaler.transform(patient_data)

    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)

    risk = probability[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease ({risk*100:.2f}%)")
    else:
        st.success(f"Low Risk of Heart Disease ({risk*100:.2f}%)")


# In[ ]:




