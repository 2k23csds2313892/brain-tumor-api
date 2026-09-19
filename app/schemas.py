from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[float]
