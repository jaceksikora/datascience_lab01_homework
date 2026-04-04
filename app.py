from fastapi import FastAPI

from inference import SentimentService
from model.PredictionRequest import PredictionRequest
from model.PredictionResponse import PredictionResponse

app = FastAPI()
service = SentimentService()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = service.predict(request.text)
    return PredictionResponse(prediction=result)
