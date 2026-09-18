import io
import os

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "brain_tumor.keras")
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "pituitary_tumor",
    "no_tumor",
]


class Avg2MaxPooling(keras.layers.Layer):
    def __init__(self, pool_size=3, strides=2, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.avg_pool = keras.layers.AveragePooling2D(
            pool_size=pool_size, strides=strides, padding=padding
        )
        self.max_pool = keras.layers.MaxPooling2D(
            pool_size=pool_size, strides=strides, padding=padding
        )

    def call(self, inputs):
        return self.avg_pool(inputs) - (
            self.max_pool(inputs) + self.max_pool(inputs)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config


class DepthwiseSeparableConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.dw = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding="same"
        )
        self.pw = keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=1, padding="valid"
        )
        self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.dw(inputs)
        x = self.pw(x)
        return tf.nn.relu(self.bn(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


CUSTOM_OBJECTS = {
    "Avg2MaxPooling": Avg2MaxPooling,
    "DepthwiseSeparableConv": DepthwiseSeparableConv,
}

app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0",
    description="Inference API for a 4-class Keras brain MRI classifier.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_brain_tumor_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Put brain_tumor.keras in the project root."
        )

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {exc}",
        )


@app.get("/")
def root():
    return {
        "service": "Brain Tumor Detection API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    x = preprocess_image(image_bytes)

    probabilities = model.predict(x, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 6),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 6)
            for i in range(len(CLASS_NAMES))
        },
    }
