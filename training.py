from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def train_model(model_path: str):
    texts = [
        "I love this product",
        "This is terrible",
        "Amazing experience",
        "Worst thing ever",
    ]

    labels = ["positive", "negative", "positive", "negative"]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
            ("clf", LogisticRegression()),
        ]
    )

    model.fit(texts, labels)

    joblib.dump(model, model_path)
