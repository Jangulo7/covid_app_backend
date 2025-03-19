import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="DetCOVID App", page_icon="🩺", layout="centered")

# Cache para cargar el modelo solo una vez
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/ensemble_unified_model3b.keras')
    return model

# Cargar el modelo
model = load_model()

# Definir clases
class_names = ["HEALTHY", "COVID-19", "PNEUMONIA"]

# Función de preprocesamiento de la imagen
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Función para obtener predicciones
def predict_image(image: np.ndarray):
    predictions = model.predict(image)[0]
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[class_index]
    return predicted_class, confidence

# --- Sección Hero ---
hero_image = "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?fit=crop&w=1200&q=80"
st.image(hero_image, use_container_width=True)
st.markdown("<h1 style='text-align: center;'>Tecnología predictiva para mejorar tu salud</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sección Cuerpo ---
st.markdown("## App para Detección de COVID-19")

st.markdown(
    """
    ¡Bienvenido a **DetCOVID App**! 
    
    Para utilizar la aplicación solo debes subir una radiografía pulmonar y obtendrás una predicción rápida sobre la posible presencia de COVID-19.

    ⚠️ *Esta herramienta no reemplaza el diagnóstico médico profesional y es solo para fines educativos y exploratorios.*

    **Instrucciones:**
    - Haz clic en **"Cargar imagen"** y selecciona tu radiografía.
    - Luego pulsa en **"Ver resultado"** para conocer la clasificación.

    ¿Te gustó la app? ¡Déjame tu feedback más abajo!
    """
)

uploaded_image = st.file_uploader("📁 Cargar imagen", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Radiografía cargada", use_container_width=True)

    if st.button("Ver resultado"):
        with st.spinner("Analizando la imagen..."):
            processed_image = preprocess_image(image)
            clasificacion, confianza = predict_image(processed_image)

        st.success(f"**Clasificación de la radiografía:** {clasificacion}")
        st.info(f"**Confianza:** {confianza:.2f}%")

# Sección para comentarios
st.markdown("### 💬 Feedback")
feedback_nombre = st.text_input("Tu nombre:")
feedback_comentario = st.text_area("Escribe tu comentario aquí:")

if st.button("Enviar comentario"):
    if feedback_nombre and feedback_comentario:
        st.write("¡Gracias por tu feedback! 💖")
        # Aquí puedes guardar el feedback en un archivo o base de datos
    else:
        st.warning("Por favor, completa todos los campos para enviar tu feedback.")

# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    footer_logo = "https://img.icons8.com/fluency/96/stethoscope.png"
    st.image(footer_logo, width=50)

with col2:
    st.markdown(
        f"""
        **Desarrollado por:** [Johanna](https://www.linkedin.com/in/tuperfil)

        📅 Marzo, {datetime.now().year}

        **Contacto:** [![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/tuperfil)
        """,
        unsafe_allow_html=True
    )

with col3:
    st.write(" ")


# --- Fin de la app ---