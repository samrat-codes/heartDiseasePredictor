import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --- LOAD THE CORRECT RANDOM FOREST MODEL ---
try:
    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("ERROR: `random_forest_model.pkl` not found. Please run the Jupyter Notebook to create it first.")
    st.stop()

# --- MAPPINGS FOR USER-FRIENDLY LABELS ---
cp_mapping = {0: '0: Typical Angina', 1: '1: Atypical Angina', 2: '2: Non-anginal Pain', 3: '3: Asymptomatic'}
restecg_mapping = {0: '0: Normal', 1: '1: ST-T wave abnormality', 2: '2: Left ventricular hypertrophy'}
slope_mapping = {0: '0: Upsloping', 1: '1: Flat', 2: '2: Downsloping'}
thal_mapping = {0: '0: NULL', 1: '1: Normal', 2: '2: Fixed Defect', 3: '3: Reversible Defect'}

# --- APP TITLE AND DESCRIPTION ---
st.title('❤️ Heart Disease Prediction App')
st.write("This app uses a Random Forest model to predict heart disease risk. The default values are set to a high-risk profile for demonstration.")
st.divider()

# --- USER INPUT FORM ---
with st.form("prediction_form"):
    st.header("Enter Patient Details")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', 1, 120, 63)
        sex = st.selectbox('Sex', (1, 0), format_func=lambda x: 'Male' if x == 1 else 'Female')
        cp = st.selectbox('Chest Pain Type (cp)', options=list(cp_mapping.keys()), format_func=lambda x: cp_mapping[x], index=3) # Default to Asymptomatic
        trestbps = st.number_input('Resting Blood Pressure', 80, 220, 160)
        
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 290)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (1, 0), format_func=lambda x: 'True' if x == 1 else 'False', index=0)
        restecg = st.selectbox('Resting ECG Results', options=list(restecg_mapping.keys()), format_func=lambda x: restecg_mapping[x], index=1)
        thalach = st.number_input('Max Heart Rate Achieved', 60, 220, 120)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)
        oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.2, 2.5)
        slope = st.selectbox('Slope of peak exercise ST segment', options=list(slope_mapping.keys()), format_func=lambda x: slope_mapping[x], index=2)
        ca = st.selectbox('Number of Major Vessels (ca)', (0, 1, 2, 3, 4), index=2)
        thal = st.selectbox('Thalassemia (thal)', options=list(thal_mapping.keys()), format_func=lambda x: thal_mapping[x], index=3)

    submitted = st.form_submit_button("Get Prediction")

if submitted:
    input_data = np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # NO SCALING IS NEEDED
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.divider()
    st.header("Prediction Result")
    
    # FIXED: Corrected the prediction interpretation
    if prediction[0] == 0:  # CHANGED from 1 to 0
        st.error(f'**Result: High Probability of Heart Disease**')
        st.write(f"The model is **{prediction_proba[0][0]*100:.2f}%** confident in this prediction.")
    else:
        st.success(f'**Result: Low Probability of Heart Disease**')
        st.write(f"The model is **{prediction_proba[0][1]*100:.2f}%** confident in this prediction.")
        
    st.info("Disclaimer: This is a prediction based on a machine learning model and not a medical diagnosis. Please consult a doctor.")
    
    # --- The lifestyle advice section remains the same ---
    st.divider()
    st.header("Lifestyle & Diet Recommendations")
    if prediction[0] == 0:  # HIGH-risk prediction
        st.subheader("Based on the HIGH-risk prediction, consider these steps (after consulting a doctor):")
        st.markdown("""
        *   **Diet:** Reduce intake of saturated fats, trans fats, and cholesterol. Increase fruits, vegetables, and lean proteins.
        *   **Lifestyle:** Engage in regular, moderate exercise. If you smoke, seek help to quit. Manage stress.
        *   **Action:** **Schedule an appointment with a cardiologist for a professional evaluation.**
        """)
    else:
        st.subheader("To maintain your LOW-risk status, continue with these healthy habits:")
        st.markdown("""
        *   **Diet:** Continue to eat a balanced diet. Stay hydrated.
        *   **Lifestyle:** Maintain a regular exercise routine and get regular preventive check-ups.
        """)