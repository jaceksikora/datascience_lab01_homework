import cloudpickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


def train_model():
    texts = [
        "What a great MLOps lecture",
        "This is terrible",
        "It's okay, nothing special",
        "Amazing experience",
        "Worst thing ever",
        "Not bad, not great",
    ]

    labels = [2, 0, 1, 2, 0, 1]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    model.save("models/embedding_model")
    with open("models/sentiment_model.pkl", "wb") as f:
        cloudpickle.dump(clf, f)
