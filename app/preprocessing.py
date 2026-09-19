import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for EfficientNetV2B0."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array
