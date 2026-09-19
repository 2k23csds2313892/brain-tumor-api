from pathlib import Path
import numpy as np
from app.model import load_brain_tumor_model
from app.preprocessing import preprocess_image

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "brain_tumor_efficientnet.keras"
THRESHOLD_PATH = BASE_DIR / "notumor_threshold.txt"

# Load model
model = load_brain_tumor_model(str(MODEL_PATH))

# Load threshold
with open(THRESHOLD_PATH, "r") as f:
    NOTUMOR_THRESHOLD = float(f.read().strip())

# Class mapping: 0: glioma, 1: meningioma, 2: pituitary, 3: no_tumor
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

def predict_brain_tumor(image_path):
    image_array = preprocess_image(image_path)
    probabilities = model.predict(image_array, verbose=0)[0]

    # Prediction logic
    if probabilities[3] >= NOTUMOR_THRESHOLD:
        predicted_index = 3
    else:
        predicted_index = int(np.argmax(probabilities[:3]))

    # Confidence must correspond to predicted class
    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }
