import os

import joblib

from training import train_model

MODEL_PATH = ".sentiment_model_joblib"


class SentimentService:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            print(f"Model was not found at {model_path}. Starting training...")
            train_model(model_path)
        else:
            print(f"Model was found at {model_path}. Loading model...")

        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        prediction = self.model.predict([text])[0]
        return prediction
