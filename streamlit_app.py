import numpy as np 
import pickle
import pandas as pd
import streamlit as st
import warnings

# disable warnings
warnings.filterwarnings('ignore')

classifier = pickle.load(open('trained_model.sav', 'rb'))


object_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Load label codes
with open("Label_codes.pkl", 'rb') as file:
    label_codes = pickle.load(file)


# Function to format given new data
def format_input(new_data):
    arr = pd.DataFrame({"Age":[new_data[0]], "Sex":[new_data[1]], "ChestPainType":[new_data[2]], "RestingBP":[new_data[3]], "FastingBS":[new_data[5]], "RestingECG":[new_data[6]], "MaxHR": [new_data[7]], "ExerciseAngina": [new_data[8]], "Oldpeak":[new_data[9]], "ST_Slope":[new_data[10]]})
    
    for col in object_cols:
        label_encode = label_codes[col]
        arr[col] = label_encode.transform(arr[col])
    
    return arr


# Function to predict the risk of heart disease
def prediction(new_data):
    new_data = new_data.split(',')
    arr = format_input(new_data)

    prediction = classifier.predict(arr)

    if prediction[0] == 1:
        return "High risk of heart disease"
    else:
        return "Low risk of heart disease"


def main():

    # giving the webpage a title
    st.title("Heart Failure Risk Predictor App")

    # getting input data from user
    Age = st.number_input("Age")
    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    RestingBP = st.number_input("Resting Blood Pressure")
    Cholesterol = st.number_input("Cholesterol")
    FastingBS = st.selectbox("Fasting Blood Sugar", ["0", "1"])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Max Heart Rate")
    ExerciseAngina = st.selectbox("Exercise Angina", ["N", "Y"])
    Oldpeak = st.number_input("Oldpeak")
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # diagnosing the risk of heart disease
    diagnosis = ''

    # submit button
    if st.button('Heart failure risk prediction result'):
        diagnosis = prediction(f'{Age},{Sex},{ChestPainType},{RestingBP},{Cholesterol},{FastingBS},{RestingECG},{MaxHR},{ExerciseAngina},{Oldpeak},{ST_Slope}')

    st.success(diagnosis)


if __name__ == "__main__":
    main()
