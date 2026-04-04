from fastapi import FastAPI

from src.inference.predictor import SentimentService
from app.models.PredictionRequest import PredictionRequest
from app.models.PredictionResponse import PredictionResponse

app = FastAPI()
service = SentimentService()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = service.predict(request.text)
    return PredictionResponse(prediction=result)
