import streamlit as st
import numpy as np
import pickle

# Set the title of the web app
st.title("Years of Experience to Salary Predictor")

# Load the trained model and the scaler
try:
    with open('finalized_model.pickle', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The model file 'finalized_model.pickle' was not found. Please ensure it is uploaded correctly.")
    st.stop()  # Stop execution if the model file is missing

try:
    with open('Scaler.pickle', 'rb') as file_r:
        scaler = pickle.load(file_r)
except FileNotFoundError:
    st.error("The scaler file 'Scaler.pickle' was not found. Please ensure it is uploaded correctly.")
    st.stop()  # Stop execution if the scaler file is missing

# User input for years of experience
x = st.number_input("Enter Years of Experience:", min_value=0.0, format="%.1f", help="Enter the number of years you have worked.")

# Predict the salary when the user clicks the 'Predict' button
if st.button("Predict"):
    try:
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(np.array([[x]]))
        # Predict the salary using the trained model
        y = model.predict(scaled_data)
        st.success(f"Predicted Salary: ${y[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
