import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

# Load Model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("model.pkl file not found!")
    st.stop()

# Title
st.title("🛒 Customer Purchase Prediction")
st.write("Enter customer details to predict whether the customer will buy the product.")

# Input Fields
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

salary = st.number_input(
    "Salary",
    min_value=0,
    value=50000
)

# Predict Button
if st.button("Predict"):

    input_data = np.array([[age, salary]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Customer will BUY the product.")
    else:
        st.error("❌ Customer will NOT BUY the product.")

# Footer
st.markdown("---")
st.caption("Machine Learning Project - Logistic Regression")