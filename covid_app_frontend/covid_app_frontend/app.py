import streamlit as st
from PIL import Image
import requests
from datetime import datetime

import streamlit as st
st.cache_data.clear()

st.set_page_config(page_title="DetCOVID App v2", page_icon="🩺", layout="centered")

# Hero
hero_image = "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?fit=crop&w=1200&q=80"
st.image(hero_image, use_column_width=True)
st.markdown("<h1 style='text-align: center;'>Predictive Technology for Health Improvement</h1>", unsafe_allow_html=True)
st.markdown("---")

# Cuerpo
st.markdown("## COVID-19 Detection App v2")

uploaded_image = st.file_uploader("📁 Upload chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict Result"):
        with st.spinner("Analyzing the image..."):
            url_backend = "https://covid-backend-x534.onrender.com"
            files = {"file": uploaded_image.getvalue()}
            response = requests.post(url_backend, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"**Classification:** {result['classification']}")
                st.info(f"**Confidence:** {result['confidence']}")
            else:
                st.error("Error fetching prediction from the backend.")




# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1,3,1])

with col1:
    footer_logo = "https://img.icons8.com/fluency/96/stethoscope.png"
    st.image(footer_logo, width=50)

with col2:
    st.markdown(
        f"""
        **Developed by:** [Your Name](https://www.linkedin.com/in/yourprofile)  
        📅 March, {datetime.now().year}  
        **Contact:** [![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/yourprofile)
        """,
        unsafe_allow_html=True
    )

with col3:
    st.write(" ")

