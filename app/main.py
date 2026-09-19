from fastapi import FastAPI, UploadFile, File
from app.predictor import predict_brain_tumor
from app.schemas import PredictionResponse
import shutil
import os

app = FastAPI(title="Brain Tumor Classification API")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_brain_tumor(temp_file)
        return result
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
