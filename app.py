import streamlit as st
import pandas as pd
import joblib # Changed from pickle to joblib
import numpy as np

# --- Load the Model ---
# Ensure 'modelo_desercion.pkl' is in the same directory as this script
def load_model():
    try:
        with open('modelo_desercion.pkl', 'rb') as file:
            model = joblib.load(file) # Changed from pickle.load to joblib.load
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.exception(e)
        return None

# Load the model directly
model = load_model()

# If model loading failed, display an error and stop the app
if model is None:
    st.error("ERROR CRÍTICO: No se pudo cargar el modelo. La aplicación no puede continuar.")
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Predicción de Deserción Estudiantil", layout="centered")

st.title("🎓 Aplicación de Predicción de Deserción Estudiantil")
st.markdown("Esta aplicación predice la probabilidad de que un estudiante deserte basándose en diversas características.")

st.sidebar.header("Parámetros de Entrada")

# --- Input Features (Now aligned with the model's expected features) ---
# Based on the latest error, the model expects:
# 'edad', 'promedio', 'trabaja', 'uso_plataforma'

# 'edad' seems to be correct, keeping it.
age = st.sidebar.slider("Edad del Estudiante", 18, 60, 20)

# New: 'promedio' (e.g., GPA or average grade)
promedio = st.sidebar.slider("Promedio Académico", 0.0, 5.0, 3.5, 0.1)

# New: 'trabaja' (binary: 0 for No, 1 for Yes)
trabaja = st.sidebar.selectbox("¿El estudiante trabaja?", ['No', 'Sí'])
trabaja_mapping = {'No': 0, 'Sí': 1}

# New: 'uso_plataforma' (e.g., hours per day/week on platform)
uso_plataforma = st.sidebar.slider("Horas de Uso de la Plataforma Semanal", 0, 40, 10)

# Create a dictionary for the input features
input_data = {
    'edad': age,
    'promedio': promedio,
    'trabaja': trabaja_mapping[trabaja],
    'uso_plataforma': uso_plataforma
}

# Convert input data to a Pandas DataFrame
# Ensure the column names exactly match the feature names used during model training
features_df = pd.DataFrame([input_data])

st.subheader("Datos de Entrada:")
st.write(features_df)

# --- Prediction ---
if st.sidebar.button("Realizar Predicción"):
    try:
        prediction_proba = model.predict_proba(features_df)[:, 1] # Probability of desertion
        prediction = model.predict(features_df)[0]

        st.subheader("Resultado de la Predicción:")
        st.write(f"Probabilidad de Deserción: **{prediction_proba[0]:.2%}**")

        if prediction == 1: # Assuming 1 means desertion, 0 means no desertion
            st.error("¡ALTO RIESGO DE DESERCIÓN! 🚨")
            st.write("Este estudiante tiene una alta probabilidad de desertar. Se recomienda una intervención temprana.")
        else:
            st.success("Bajo Riesgo de Deserción. 👍")
            st.write("Este estudiante tiene una baja probabilidad de desertar.")

    except Exception as e:
        st.error(f"Ha ocurrido un error durante la predicción: {e}")
        st.write("Por favor, asegúrese de que los nombres y tipos de las variables de entrada coincidan con los esperados por el modelo.")

st.markdown("--- ")
st.markdown("Desarrollado con ❤️ por Colab Composer")
