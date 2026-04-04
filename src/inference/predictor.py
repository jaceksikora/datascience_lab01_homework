import os
import cloudpickle
from sentence_transformers import SentenceTransformer
from src.training.train import train_model

MODEL_PATH = "models/sentiment_model.pkl"
EMBEDDING_MODEL_PATH = "models/embedding_model"

label_map = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentService:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path) or not os.path.exists(EMBEDDING_MODEL_PATH):
            print("Model not found starting training..")
            os.makedirs("models", exist_ok=True)
            train_model()
            print("Training completed.")

        with open(model_path, "rb") as f:
            self.classifier = cloudpickle.load(f)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def predict(self, text: str) -> str:
        embedding = self.encoder.encode([text])
        prediction = self.classifier.predict(embedding)
        return label_map[prediction[0]]
