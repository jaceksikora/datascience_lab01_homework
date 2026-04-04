from fastapi import FastAPI

from app.schemas.PredictionRequest import PredictionRequest
from app.schemas.PredictionResponse import PredictionResponse
from src.inference.predictor import SentimentService

app = FastAPI()
service = SentimentService()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = service.predict(request.text)
    return PredictionResponse(prediction=result)
