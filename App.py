import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🩺"
)

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved models
diabetes_model = pickle.load(open(f"{working_dir}/saved_models/diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open(f"{working_dir}/saved_models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open(f"{working_dir}/saved_models/parkinsons_model.sav", "rb"))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "person"],
        menu_icon="hospital-fill",
        default_index=0
    )

# Helper function for user input validation
def validate_inputs(inputs):
    try:
        return [float(x) for x in inputs]
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
        return None

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using Machine Learning")
    st.markdown("### Please provide the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", help="Enter the number of pregnancies.")
    with col2:
        Glucose = st.text_input("Glucose Level", help="Enter the glucose level in mg/dL.")
    with col3:
        BloodPressure = st.text_input("Blood Pressure", help="Enter blood pressure in mmHg.")
    with col1:
        SkinThickness = st.text_input("Skin Thickness", help="Enter skin fold thickness in mm.")
    with col2:
        Insulin = st.text_input("Insulin Level", help="Enter insulin level in uU/mL.")
    with col3:
        BMI = st.text_input("BMI", help="Enter body mass index (weight/height^2).")
    with col1:
        DiabetesPedigreeFunction = st.text_input(
            "Diabetes Pedigree Function", help="Enter a measure of diabetes history in relatives."
        )
    with col2:
        Age = st.text_input("Age", help="Enter the age of the person.")

    if st.button("Predict Diabetes"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        validated_input = validate_inputs(user_input)

        if validated_input:
            prediction = diabetes_model.predict([validated_input])[0]
            result = "The person is diabetic." if prediction == 1 else "The person is not diabetic."
            st.success(result)

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using Machine Learning")
    st.markdown("### Please provide the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age", help="Enter the age of the person.")
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"], help="Select the gender.")
        sex = 1 if sex == "Male" else 0
    with col3:
        cp = st.text_input("Chest Pain Types", help="Enter chest pain type (0-3).")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure", help="Enter resting blood pressure in mmHg.")
    with col2:
        chol = st.text_input("Serum Cholesterol", help="Enter serum cholesterol in mg/dL.")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar", help="1 if > 120 mg/dL, else 0.")
    with col1:
        restecg = st.text_input("Resting ECG", help="Enter resting ECG results (0-2).")
    with col2:
        thalach = st.text_input("Max Heart Rate", help="Enter maximum heart rate achieved.")
    with col3:
        exang = st.text_input("Exercise Induced Angina", help="1 if yes, else 0.")
    with col1:
        oldpeak = st.text_input("ST Depression", help="Enter ST depression value.")
    with col2:
        slope = st.text_input("Slope of ST Segment", help="Enter slope (0-2).")
    with col3:
        ca = st.text_input("Major Vessels", help="Enter the number of major vessels (0-4).")
    with col1:
        thal = st.text_input("Thalassemia", help="Enter 0 (normal), 1 (fixed defect), or 2 (reversible defect).")

    if st.button("Predict Heart Disease"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        validated_input = validate_inputs(user_input)

        if validated_input:
            prediction = heart_disease_model.predict([validated_input])[0]
            result = "The person has heart disease." if prediction == 1 else "The person does not have heart disease."
            st.success(result)

# Parkinson's Prediction
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using Machine Learning")
    st.markdown("### Please provide the following details:")

    input_fields = {
        "MDVP:Fo(Hz)": "Enter average vocal fundamental frequency.",
        "MDVP:Fhi(Hz)": "Enter highest vocal fundamental frequency.",
        "MDVP:Flo(Hz)": "Enter lowest vocal fundamental frequency.",
        "MDVP:Jitter(%)": "Enter jitter percentage.",
        "MDVP:Jitter(Abs)": "Enter absolute jitter.",
        "MDVP:RAP": "Enter relative amplitude perturbation.",
        "MDVP:PPQ": "Enter pitch perturbation quotient.",
        "Jitter:DDP": "Enter DDP jitter.",
        "MDVP:Shimmer": "Enter shimmer.",
        "MDVP:Shimmer(dB)": "Enter shimmer in dB.",
        "Shimmer:APQ3": "Enter APQ3 shimmer.",
        "Shimmer:APQ5": "Enter APQ5 shimmer.",
        "MDVP:APQ": "Enter APQ shimmer.",
        "Shimmer:DDA": "Enter DDA shimmer.",
        "NHR": "Enter noise-to-harmonics ratio.",
        "HNR": "Enter harmonics-to-noise ratio.",
        "RPDE": "Enter recurrence period density entropy.",
        "DFA": "Enter detrended fluctuation analysis.",
        "spread1": "Enter spread1.",
        "spread2": "Enter spread2.",
        "D2": "Enter D2.",
        "PPE": "Enter PPE.",
    }

    user_input = {}
    for field, tooltip in input_fields.items():
        user_input[field] = st.text_input(field, help=tooltip)

    if st.button("Predict Parkinson's Disease"):
        validated_input = validate_inputs(list(user_input.values()))

        if validated_input:
            prediction = parkinsons_model.predict([validated_input])[0]
            result = "The person has Parkinson's disease." if prediction == 1 else "The person does not have Parkinson's disease."
            st.success(result)
