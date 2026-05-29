
import streamlit as st
import numpy as np
import pickle

loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

st.title('Diabetes Prediction Web App')

Pregnancies = st.text_input('Pregnancies')
Glucose = st.text_input('Glucose')
BloodPressure = st.text_input('Blood Pressure')
SkinThickness = st.text_input('Skin Thickness')
Insulin = st.text_input('Insulin')
BMI = st.text_input('BMI')
DiabetesPedigreeFunction = st.text_input('DPF')
Age = st.text_input('Age')

if st.button('Test Result'):

    input_data = (
        float(Pregnancies),
        float(Glucose),
        float(BloodPressure),
        float(SkinThickness),
        float(Insulin),
        float(BMI),
        float(DiabetesPedigreeFunction),
        float(Age)
    )

    input_array = np.asarray(input_data)
    input_reshaped = input_array.reshape(1, -1)

    prediction = loaded_model.predict(input_reshaped)

    if prediction[0] == 0:
        st.success('Not Diabetic')
    else:
        st.success('Diabetic')
