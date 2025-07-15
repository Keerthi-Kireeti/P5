import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🩺"
)

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/saved_models"
os.makedirs(model_path, exist_ok=True)

# Function to train and save a model
def train_and_save_model(dataset_path, model_filename):
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns='Outcome', axis=1)
    Y = dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    pickle.dump(classifier, open(f"{model_path}/{model_filename}", 'wb'))
    return classifier

# Load or train models
def load_model(model_filename, dataset_path):
    model_file = f"{model_path}/{model_filename}"
    if os.path.exists(model_file):
        return pickle.load(open(model_file, 'rb'))
    else:
        return train_and_save_model(dataset_path, model_filename)

diabetes_model = load_model("diabetes_model.sav", "diabetes.csv")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Disease Prediction System",
        ["Diabetes Prediction"],
        icons=["activity"],
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
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure")
    with col1:
        SkinThickness = st.text_input("Skin Thickness")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2:
        Age = st.text_input("Age")

    if st.button("Predict Diabetes"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        validated_input = validate_inputs(user_input)
        
        if validated_input:
            prediction = diabetes_model.predict([validated_input])[0]
            result = "The person is diabetic." if prediction == 1 else "The person is not diabetic."
            st.success(result)
