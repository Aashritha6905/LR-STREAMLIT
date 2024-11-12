# Streamlit app for deploying logistic regression model

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit title and description
st.title("Logistic Regression Prediction App")
st.write("Enter the feature values to predict the target")

# Define feature inputs for the app
# Add inputs for each feature
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# Add more inputs based on your dataset's features...

# Make a prediction based on user input
input_data = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])  # adjust columns as per your model
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Predicted Outcome:", prediction[0])
