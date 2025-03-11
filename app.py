import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Load the trained scaler (important!)
scaler = joblib.load('scaler.pkl')

# Streamlit app interface
st.title("Student Exam Score Prediction")
st.markdown("Enter the values below to predict the exam score:")

# Create input fields
hours_studied = st.number_input("Hours Studied", min_value=0)
attendance = st.number_input("Attendance", min_value=0)
sleep_hours = st.number_input("Sleep Hours", min_value=0)
previous_scores = st.number_input("Previous Scores", min_value=0)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0)
physical_activity = st.number_input("Physical Activity", min_value=0)

# Create new meaningful features
study_efficiency = hours_studied / (attendance + 1)  # Avoid division by zero
improvement_rate = 0  # Placeholder
tutoring_effect = tutoring_sessions / (hours_studied + 1)

# Prepare input data
input_data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep_hours],
    'Previous_Scores': [previous_scores],
    'Tutoring_Sessions': [tutoring_sessions],
    'Physical_Activity': [physical_activity],
    'Study_Efficiency': [study_efficiency],
    'Improvement_Rate': [improvement_rate],
    'Tutoring_Effect': [tutoring_effect]
})

# Ensure feature names match
input_data = input_data[model.feature_names_in_]

# Apply the trained scaler (FIXED!)
input_data_scaled = scaler.transform(input_data)

# Button to make a prediction
if st.button("Predict Exam Score"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Exam Score: {prediction[0]:.2f}")
