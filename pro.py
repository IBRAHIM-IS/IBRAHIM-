import streamlit as st
import numpy as np
import pickle
import os

st.title("Years of Experience to Salary Predictor")

try:
    with open('finalized_model.pickle', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The model file 'finalized_model.pickle' was not found. Please upload the file or check the file path.")
    print("Current working directory:", os.getcwd())
    print("Files in the current directory:", os.listdir())
with open('Scaler.pickle', 'rb') as file_r:
        scaler = pickle.load(file_r)
x = st.number_input("Enter Years of Experience:", min_value=0.0, format="%.1f")

if st.button("Predict"):
    scaled_data = scaler.transform(np.array([[x]]))

    y = model.predict(scaled_data)
    st.write(f"Predicted Salary: ${y[0]:.2f}")