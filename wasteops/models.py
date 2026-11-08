from dataclasses import dataclass
from typing import Protocol
from PIL import Image

@dataclass
class Prediction:
    label: str
    score: float
    alternatives: list[dict]
    backend: str
    abstained: bool = False
    note: str = "Scores are model scores, not calibrated probabilities."

class WasteModel(Protocol):
    def predict(self, image: Image.Image) -> Prediction: ...
