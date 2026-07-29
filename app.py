import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="HeartWise AI",
    page_icon="❤️",
    layout="wide"
)

# Load saved model and feature names
try:
    model = joblib.load("heart_model.pkl")
    feature_names = joblib.load("feature_names.pkl")

except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()

# Title
st.title("❤️ HeartWise AI")
st.subheader("Heart Disease Risk Prediction")

st.write(
    "Enter the patient's medical information below. "
    "The model will estimate whether the patient may be at risk of heart disease."
)

st.warning(
    "This application is for educational purposes only and should not replace professional medical advice."
)

# Sidebar
with st.sidebar:
    st.header("About the Project")
    st.write(
        "HeartWise AI uses machine learning to predict heart disease risk "
        "based on patient health information."
    )

    st.write("**Model used:** Tuned Random Forest")
    st.write("**Task:** Binary classification")
    st.write("**Output:** Heart disease risk or no heart disease risk")

# Input form
with st.form("patient_form"):

    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:
        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar above 120 mg/dl",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    with col3:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    predict_button = st.form_submit_button("Predict Heart Disease Risk")

# Prediction
if predict_button:

    patient_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    # Apply the same one-hot encoding used during training
    patient_encoded = pd.get_dummies(patient_data)

    # Match the exact feature columns used by the model
    patient_encoded = patient_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

try:
    prediction = model.predict(patient_encoded)[0]
    probability = model.predict_proba(patient_encoded)[0]

except Exception:
    st.error("Prediction failed. Please try again.")
    st.stop()

    st.divider()
    st.header("Prediction Result")

    if prediction == 1:
        confidence = probability[1] * 100

        st.error("High Risk of Heart Disease")
        st.metric(
            "Prediction Confidence",
            f"{confidence:.1f}%"
        )

        st.write(
            "The model predicts that this patient may have a higher risk "
            "of heart disease. The patient should consider consulting a "
            "qualified healthcare professional for further assessment."
        )

    else:
        confidence = probability[0] * 100

        st.success("Low Risk of Heart Disease")
        st.metric(
            "Prediction Confidence",
            f"{confidence:.1f}%"
        )

        st.write(
            "The model predicts that this patient has a lower risk of "
            "heart disease based on the entered information. Maintaining "
            "regular health checks and a healthy lifestyle is still recommended."
        )

    with st.expander("View Encoded Model Input"):
        st.dataframe(patient_encoded)