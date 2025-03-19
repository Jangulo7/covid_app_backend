from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Inicializar la aplicación FastAPI
app = FastAPI(title="COVID-19 Classification API")

# Configurar CORS
origins = [
    "https://tu-frontend-en-streamlit.com",  # Reemplaza con la URL de tu frontend
    "https://otro-origen-permitido.com",     # Añade otros orígenes permitidos según sea necesario
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permite solicitudes solo de los orígenes especificados
    allow_credentials=True,
    allow_methods=["*"],    # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],    # Permite todas las cabeceras
)

# Load quantized model TFLite
interpreter = tf.lite.Interpreter(model_path="model/unified_ensemble_model3b_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["HEALTHY", "COVID-19", "PNEUMONIA"]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return class_names[class_index], confidence

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(content={"error": "Invalid file format."}, status_code=400)
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    processed_image = preprocess_image(image)
    predicted_class, confidence = predict(processed_image)

    return {
        "classification": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }
