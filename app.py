import streamlit as st
import joblib
import numpy as np
import pandas



# load the trained model
model = joblib.load("final_diabetes_model")

# Title
st.title("Diabetes Prediction App")

st.write("This App predicts if someone has diabetes or not")


# Input FORM
glucose = st.number_input("Enter Glucose",min_value=0.0,max_value=200.0,step=0.1)
bp = st.number_input("Enter Blood Pressure",min_value=0.0,max_value=100.0,step=0.1)
skin = st.number_input("Enter Thickness",min_value=0.0,max_value=100.0,step=0.1)
insulin = st.number_input("Enter Insulin",min_value=0.0,max_value=200.0,step=0.1)
bmi = st.number_input("Enter BMI",min_value=0.0,max_value=100.0,step=0.1)
dpf = st.number_input("Enter Diabetes Pedigree Function",min_value=0.0,max_value=100.0,step=0.1)
age = st.number_input("Enter Age",min_value=0.0,max_value=100.0,step=0.1)

if st.button("Predict Diabetes"):
    input_data = np.array([[glucose,bp,skin,insulin,bmi,dpf,age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error(f"This Patient Has Diabetes")
    else:
        st.success(f"This Patient Does not have Diabetes")