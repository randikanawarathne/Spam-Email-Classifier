"""Evaluation module for spam email classification."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

from .preprocess import preprocess_email
from .train import load_dataset


def evaluate_on_test_set(model_path: str, data_path: str) -> dict:
    """Evaluate model on a test set.

    Args:
        model_path: Path to the saved model
        data_path: Path to the dataset CSV

    Returns:
        Dictionary of evaluation metrics
    """
    # Load model and data
    model = joblib.load(model_path)
    df = load_dataset(data_path)

    # Preprocess and predict
    df['processed_text'] = df['text'].apply(preprocess_email)
    y_true = df['label']
    y_pred = model.predict(df['processed_text'])

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_spam': precision_score(y_true, y_pred, pos_label='spam'),
        'recall_spam': recall_score(y_true, y_pred, pos_label='spam'),
        'f1_spam': f1_score(y_true, y_pred, pos_label='spam'),
    }

    return metrics


def print_evaluation_report(model_path: str, data_path: str) -> None:
    """Print detailed evaluation report.

    Args:
        model_path: Path to the saved model
        data_path: Path to the dataset CSV
    """
    model = joblib.load(model_path)
    df = load_dataset(data_path)

    df['processed_text'] = df['text'].apply(preprocess_email)
    y_true = df['label']
    y_pred = model.predict(df['processed_text'])

    print("=" * 50)
    print("SPAM EMAIL CLASSIFIER - EVALUATION REPORT")
    print("=" * 50)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
    print(f"              Predicted")
    print(f"              ham    spam")
    print(f"Actual ham    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual spam   {cm[1][0]:4d}    {cm[1][1]:4d}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nOverall Metrics:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (spam): {precision_score(y_true, y_pred, pos_label='spam'):.4f}")
    print(f"Recall (spam):    {recall_score(y_true, y_pred, pos_label='spam'):.4f}")
    print(f"F1 Score (spam):   {f1_score(y_true, y_pred, pos_label='spam'):.4f}")


if __name__ == "__main__":
    print("Evaluation module - use main.py to evaluate model")
