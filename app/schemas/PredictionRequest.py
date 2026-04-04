from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, val: str) -> str:
        if not val.strip():
            raise ValueError("Text cannot be empty")
        return val
