from src.inference.predictor import SentimentService


def test_model_loading():
    service = SentimentService()
    assert service.classifier is not None
    assert service.encoder is not None
