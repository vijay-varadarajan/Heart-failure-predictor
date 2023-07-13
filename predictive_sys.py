import numpy as np
import pickle
import pandas as pd
import warnings

# disable warnings
warnings.filterwarnings('ignore')

classifier = pickle.load(open('C:/Users/varad/Documents/ML_projects/Tabular classification/Heart failure predictor/trained_model.sav', 'rb'))

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

def prediction(new_data):
    new_data = new_data.split(',')
    arr = format_input(new_data)

    prediction = classifier.predict(arr)

    if prediction[0] == 1:
        return ("High risk of heart disease")
    else:
        return ("Low risk of heart disease")

print(prediction('49,F,ATA,110,208,0,Normal,160,N,0,Up'))
print(prediction('57,M,ASY,150,255,0,Normal,92,Y,3,Flat'))
