# Sentiment Analysis Service

This project provides a simple Sentiment Analysis API built with **FastAPI**, **Sentence Transformers**, and **Scikit-learn**. It classifies input text as positive, neutral, or negative.

## Features
- **FastAPI** for providing a RESTful API endpoint.
- **Sentence-BERT** (`all-MiniLM-L6-v2`) for generating high-quality text embeddings.
- **Logistic Regression** for sentiment classification.
- **Dockerized** setup for easy deployment and reproduction.

## Quick Start

### Using Docker Compose (Recommended)
The service and its dependencies are containerized. To start the API, run:
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

> **Note:** The first request (or startup) might take longer because the system needs to download the Sentence-Transformer model and train the classifier if the `models/` directory is empty.

### API Usage
Once the service is running, you can send POST requests to the `/predict` endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "What a great MLOps lecture"}'
```

**Response:**
```json
{"prediction": "positive"}
```

## Project Structure
- `app/`: FastAPI application and schemas
- `src/training/`: Scripts for training the sentiment classifier
- `src/inference/`: Prediction logic and model loading
- `models/`: Directory containing saved model weights (automatically generated if missing)
- `tests/`: Unit tests for model and API functionality