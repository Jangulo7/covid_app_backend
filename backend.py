from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
import os

# --------------------------------------------
# NUEVO: Configurar variables para reducir CPU/RAM uso
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# --------------------------------------------

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Inicializar la aplicación FastAPI
app = FastAPI(title="COVID-19 Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],    # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],    # Permite todas las cabeceras
)

# ------------------------------
# Cargar modelo TFLite DESPUÉS de ajustar las opciones
interpreter = tf.lite.Interpreter(
    model_path="model/unified_ensemble_model3b_quantized.tflite",
    experimental_delegates=[]  # Desactiva XNNPACK para evitar uso excesivo
)
interpreter.allocate_tensors()
# ------------------------------

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
    try:
        logging.info(f"Received file: {file.filename}, type: {file.content_type}")
        
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            logging.error("Invalid file format.")
            return JSONResponse(content={"error": "Invalid file format."}, status_code=400)
        
        image_data = await file.read()
        logging.info(f"File size: {len(image_data)} bytes")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logging.info("Image successfully opened and converted.")

        processed_image = preprocess_image(image)
        predicted_class, confidence = predict(processed_image)

        logging.info(f"Prediction done: {predicted_class}, confidence: {confidence:.2f}%")
        
        return {
            "classification": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)
