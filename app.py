import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('disease_model.pkl', 'rb'))

# Load dataset and symptoms
data = pd.read_csv('Training.csv')
symptoms = [col.strip() for col in data.columns if col.strip().lower() != 'prognosis']

# Custom CSS styling
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 0;
    }
    .description {
        font-size: 20px;
        color: #3a7ca5;
        text-align: center;
        margin-top: 0;
        margin-bottom: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div.stButton > button {
        background-color: #256d85;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1b4f61;
        cursor: pointer;
    }
    .css-1wy0on6 {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown('<h1 class="title">Medical Diagnosis Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Select your symptoms and get a possible diagnosis</p>', unsafe_allow_html=True)

# Symptom selection
selected_symptoms = st.multiselect("Select your symptoms:", symptoms)

# Diagnose
if st.button("Diagnose"):
    if not selected_symptoms:
        st.error("⚠️ Please select at least one symptom.")
    else:
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_data[symptoms.index(symptom)] = 1

        # Predict
        prediction = model.predict([input_data])
        st.success(f"🩺 Possible disease: **{prediction[0]}**")
