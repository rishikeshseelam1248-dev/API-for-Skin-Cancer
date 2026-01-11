from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["https://69628d499fde5ab0388b0f3a--darling-youtiao-37f7ea.netlify.app/"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

MODEL = keras.models.load_model("/Users/srilathayatham/Downloads/Skin_Cancer_Classification/8.keras")
class_names = ["Benign", "Malignant Skin Cancer", "Normal"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = (math.floor(np.max(predictions[0]) * 1000)) / 1000
    return {
        'class': predicted_class,
        'confidence': str(confidence * 100)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
 