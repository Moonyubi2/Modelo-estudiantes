import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load the Model ---
def load_model():
    try:
        with open('modelo_desercion.pkl', 'rb') as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.exception(e)
        return None

model = load_model()

if model is None:
    st.error("ERROR CRÍTICO: No se pudo cargar el modelo. La aplicación no puede continuar.")
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Predicción de Deserción Estudiantil", layout="centered")

st.title("🎓 Aplicación de Predicción de Deserción Estudiantil")
st.markdown("Esta aplicación predice la probabilidad de que un estudiante deserte basándose en diversas características.")

st.sidebar.header("Parámetros de Entrada")

# --- Input Features ---
# Existing features from previous iterations
age = st.sidebar.slider("Edad del Estudiante", 18, 60, 20)
promedio = st.sidebar.slider("Promedio Académico", 0.0, 5.0, 3.5, 0.1)

trabaja = st.sidebar.selectbox("¿El estudiante trabaja?", ['No', 'Sí'])
trabaja_mapping = {'No': 0, 'Sí': 1}

uso_plataforma = st.sidebar.slider("Horas de Uso de la Plataforma Semanal", 0, 40, 10)

# NEW FEATURES REQUESTED BY USER
acceso_internet = st.sidebar.selectbox("Acceso a Internet", ['Sí', 'No'])
acceso_internet_mapping = {'Sí': 1, 'No': 0}

asistencia = st.sidebar.slider("Porcentaje de Asistencia", 0, 100, 80)

horas_estudio = st.sidebar.slider("Horas de Estudio Semanales", 0, 50, 15)

materias_perdidas = st.sidebar.slider("Número de Materias Perdidas", 0, 10, 0)

nivel_socioeconomico = st.sidebar.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
nivel_socioeconomico_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}


# Create a dictionary for the input features
input_data = {
    'edad': age,
    'promedio': promedio,
    'trabaja': trabaja_mapping[trabaja],
    'uso_plataforma': uso_plataforma,
    'acceso_internet': acceso_internet_mapping[acceso_internet],
    'asistencia': asistencia,
    'horas_estudio': horas_estudio,
    'materias_perdidas': materias_perdidas,
    'nivel_socioeconomico': nivel_socioeconomico_mapping[nivel_socioeconomico]
}

# Convert input data to a Pandas DataFrame
features_df = pd.DataFrame([input_data])

st.subheader("Datos de Entrada:")
st.write(features_df)

# --- Prediction ---
if st.sidebar.button("Realizar Predicción"):
    try:
        prediction_proba = model.predict_proba(features_df)[:, 1]
        prediction = model.predict(features_df)[0]

        st.subheader("Resultado de la Predicción:")
        st.write(f"Probabilidad de Deserción: **{prediction_proba[0]:.2%}**")

        if prediction == 1:
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
