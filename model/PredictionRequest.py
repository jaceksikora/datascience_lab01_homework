from pydantic import BaseModel


class PredictionRequest(BaseModel):
    prediction: str
