import joblib
from sentence_transformers import SentenceTransformer


MODEL_PATH = "models/sentiment_model.joblib"

label_map = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentService:
    def __init__(self, model_path: str = "models/sentiment_model.joblib"):
        self.classifier = joblib.load(model_path)
        self.encoder = SentenceTransformer("models/embedding_model")

    def predict(self, text: str) -> str:
        embedding = self.encoder.encode([text])
        prediction = self.classifier.predict(embedding)
        return label_map[prediction[0]]
