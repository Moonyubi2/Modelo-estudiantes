import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load the Model ---
# Ensure 'modelo_desercion.pkl' is in the same directory as this script
def load_model():
    try:
        with open('modelo_desercion.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Instead of st.error and st.stop here, return None and handle outside
        return None
    except Exception as e:
        # Use st.exception for detailed error reporting in Streamlit
        st.exception(e)
        return None

# Load the model directly
model = load_model()

# If model loading failed, display an error and stop the app
if model is None:
    st.error("ERROR CRÍTICO: No se pudo cargar el modelo. La aplicación no puede continuar.")
    st.stop() # This st.stop will now be called after initial Streamlit setup

# --- Streamlit App Layout ---
st.set_page_config(page_title="Predicción de Deserción Estudiantil", layout="centered")

st.title("🎓 Aplicación de Predicción de Deserción Estudiantil")
st.markdown("Esta aplicación predice la probabilidad de que un estudiante deserte basándose en diversas características.")

st.sidebar.header("Parámetros de Entrada")

# --- Input Features (Customize these based on your model's exact features) ---
# Example: Sliders for numerical features
age = st.sidebar.slider("Edad del Estudiante", 18, 60, 20)
academic_performance = st.sidebar.slider("Rendimiento Académico (GPA promedio)", 0.0, 5.0, 3.5, 0.1)
credits_enrolled = st.sidebar.slider("Créditos Matriculados por Semestre", 5, 30, 15)

# Example: Selectboxes for categorical features
gender = st.sidebar.selectbox("Género", ['Masculino', 'Femenino', 'Otro'])
financial_aid = st.sidebar.selectbox("Recibe Ayuda Financiera", ['Sí', 'No'])
previous_education = st.sidebar.selectbox("Nivel Educativo Previo", ['Bachillerato', 'Técnico', 'Licenciatura (incompleta)'])

# Map categorical inputs to numerical/encoded values as your model expects
# You MUST adjust this mapping according to your model's training data encoding
gender_mapping = {'Masculino': 0, 'Femenino': 1, 'Otro': 2}
financial_aid_mapping = {'Sí': 1, 'No': 0}
previous_education_mapping = {'Bachillerato': 0, 'Técnico': 1, 'Licenciatura (incompleta)': 2}

# Create a dictionary for the input features
input_data = {
    'edad': age,
    'rendimiento_academico': academic_performance,
    'creditos_matriculados': credits_enrolled,
    'genero': gender_mapping[gender],
    'ayuda_financiera': financial_aid_mapping[financial_aid],
    'educacion_previa': previous_education_mapping[previous_education],
    # Add more features here if your model requires them
    # e.g., 'ingresos_familiares': st.sidebar.slider("Ingresos Familiares", 0, 100000, 30000)
    # e.g., 'distancia_casa': st.sidebar.slider("Distancia de Casa (km)", 0, 200, 10)
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
