import tensorflow as tf

MODEL_PATH = "models/brain_tumor_efficientnet.keras"

def load_brain_tumor_model(model_path: str):
    return tf.keras.models.load_model(model_path)
