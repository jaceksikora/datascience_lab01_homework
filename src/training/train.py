import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_model():
    texts = [
        "What a great MLOps lecture, I am very satisfied",
        "This is terrible",
        "It's okay, nothing special",
        "Amazing experience",
        "Worst thing ever",
        "Not bad, not great",
    ]

    labels = [2, 0, 1, 2, 0, 1]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    X = model.encode(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test)))

    model.save("models/embedding_model")
    joblib.dump(clf, "models/sentiment_model.joblib")
