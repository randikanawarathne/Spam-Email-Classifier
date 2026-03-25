"""Prediction module for spam email classification."""

import joblib
from typing import Tuple

from .preprocess import preprocess_email


def predict_email(model, text: str) -> Tuple[str, float]:
    """Predict if an email is spam or ham.

    Args:
        model: Trained model pipeline
        text: Raw email text

    Returns:
        Tuple of (prediction, confidence)
    """
    processed_text = preprocess_email(text)

    # Get prediction
    prediction = model.predict([processed_text])[0]

    # Get confidence scores
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([processed_text])[0]
        confidence = max(probabilities)
    else:
        # For models without predict_proba (like LinearSVC)
        confidence = 1.0 if prediction == 'spam' else 1.0

    return prediction, confidence


def predict_batch(model, texts: list) -> list:
    """Predict multiple emails.

    Args:
        model: Trained model pipeline
        texts: List of raw email texts

    Returns:
        List of (prediction, confidence) tuples
    """
    processed_texts = [preprocess_email(text) for text in texts]
    predictions = model.predict(processed_texts)

    results = []
    for pred in predictions:
        results.append((pred, 1.0))

    return results


def main(model_path: str, text: str) -> None:
    """Make a prediction using a saved model.

    Args:
        model_path: Path to the saved model
        text: Email text to classify
    """
    model = joblib.load(model_path)
    prediction, confidence = predict_email(model, text)

    print(f"\nEmail preview: {text[:80]}...")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    print("Prediction module - use main.py to make predictions")
