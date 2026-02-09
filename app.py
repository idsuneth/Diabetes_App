import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and feature structure
model = joblib.load("rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# ---- UI HEADER ----
st.title("🩺 Diabetes Risk Prediction")
st.write("This application predicts diabetes risk using a trained machine learning model.")

st.divider()

# ---- USER INPUTS ----
st.subheader("Enter Patient Information")

age = st.number_input("Age", min_value=1, max_value=100, value=35)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=120, value=80)

st.divider()

# ---- CREATE FULL FEATURE VECTOR ----
def create_input_dataframe():
    # Start with all features = 0
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Fill known values
    if "age" in input_data.columns:
        input_data["age"] = age
    if "bmi" in input_data.columns:
        input_data["bmi"] = bmi
    if "systolic_bp" in input_data.columns:
        input_data["systolic_bp"] = systolic_bp
    if "diastolic_bp" in input_data.columns:
        input_data["diastolic_bp"] = diastolic_bp

    return input_data

# ---- PREDICTION ----
if st.button("🔍 Predict Diabetes Risk"):
    input_df = create_input_dataframe()
    prediction = model.predict(input_df)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High risk of diabetes detected")
    else:
        st.success("✅ Low risk of diabetes detected")

st.caption("⚕️ This tool is for academic demonstration purposes only.")
