from fastapi import FastAPI

from model.PredictionResponse import PredictionResponse

app = FastAPI()


@app.post("/predict")
def predict(request: PredictionResponse) -> dict[str, str]:
    return {"status": request.prediction}
