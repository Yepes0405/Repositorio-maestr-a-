
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration --- #
st.set_page_config(
    page_title="Iris Species Predictor 🌸",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling (Color Theory & UX) ---
st.markdown('''
<style>
    .main-header {color: #4CAF50; font-size: 2.5em; text-align: center; margin-bottom: 20px;}
    .subheader {color: #333333; font-size: 1.2em; text-align: center; margin-bottom: 30px;}
    .stButton>button {background-color: #9C27B0; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-size: 1.1em; margin-top: 20px;}
    .stButton>button:hover {background-color: #7B1FA2; color: white;}
    .prediction-box {background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 8px; margin-top: 30px;}
    .prediction-text {color: #1B5E20; font-size: 1.5em; font-weight: bold;}
    .sidebar-header {color: #333333;}
    .stExpander div[data-baseweb="accordion-item-header"] {background-color: #F0F4C3; border-radius: 5px;}
</style>
''', unsafe_allow_html=True)

# --- Load Models --- #
try:
    with open('knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Error: `knn_model.pkl` not found. Please ensure it's in the correct directory.")
    st.stop()

try:
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Error: `label_encoder.pkl` not found. Please ensure it's in the correct directory.")
    st.stop()

# --- Main Application Content --- #
st.markdown('<h1 class="main-header">🌸 Iris Species Predictor 🌸</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Input the measurements of the Iris flower below to get an instant species prediction.</p>', unsafe_allow_html=True)

st.divider()

st.subheader("📊 Enter Flower Measurements:")

# Using columns for better input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("📏 Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider("↔️ Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)

with col2:
    petal_length = st.slider("📏 Petal Length (cm)", 1.0, 7.0, 1.3, 0.1)
    petal_width = st.slider("↔️ Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


if st.button("✨ Predict Species!"):
    try:
        prediction_raw = knn_model.predict(input_data)
        predicted_species = label_encoder.inverse_transform(prediction_raw)

        st.markdown('''
        <div class="prediction-box">
            <p class="prediction-text">Predicted Species: <b>{}</b></p>
        </div>
        '''.format(predicted_species[0]), unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.divider()

# --- About Section (using st.expander) --- #
with st.expander("ℹ️ About This App"): # Using st.expander for cleaner UI
    st.markdown(
        "This interactive web application utilizes a **K-Nearest Neighbors (KNN)** machine learning model "
        "to classify Iris flower species. The model, along with a **Label Encoder**, "
        "was pre-trained and loaded from `.pkl` files. Simply adjust the sliders "
        "to input the sepal and petal measurements, and click 'Predict Species!' to see the result. "
        "Built with Streamlit for a delightful user experience!"
    )
    st.markdown("**Model:** K-Nearest Neighbors (KNN)")
    st.markdown("**Data Source:** Iris Dataset")
    st.markdown("**Developed for:** Learning & Demonstration")

