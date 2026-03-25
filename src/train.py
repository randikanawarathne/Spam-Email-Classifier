"""Model training module for spam email classification."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from .preprocess import preprocess_email


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load email dataset from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare data for training.

    Args:
        df: DataFrame with 'text' and 'label' columns

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_email)

    # Split data
    X = df['processed_text']
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_naive_bayes(X_train, y_train) -> Pipeline:
    """Train Naive Bayes classifier.

    Args:
        X_train: Training text data
        y_train: Training labels

    Returns:
        Trained pipeline
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_svm(X_train, y_train) -> Pipeline:
    """Train SVM classifier.

    Args:
        X_train: Training text data
        y_train: Training labels

    Returns:
        Trained pipeline
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', LinearSVC(random_state=42, max_iter=10000))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test) -> None:
    """Evaluate model and print metrics.

    Args:
        pipeline: Trained pipeline
        X_test: Test text data
        y_test: Test labels
    """
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))


def save_model(pipeline, filepath: str) -> None:
    """Save trained model to disk.

    Args:
        pipeline: Trained pipeline
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Pipeline:
    """Load trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded pipeline
    """
    return joblib.load(filepath)


def train_model(data_path: str, model_path: str, model_type: str = 'nb') -> None:
    """Train and save a spam classification model.

    Args:
        data_path: Path to the dataset CSV
        model_path: Path to save the trained model
        model_type: 'nb' for Naive Bayes, 'svm' for SVM
    """
    print("Loading dataset...")
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} emails ({df['label'].value_counts().to_dict()})")

    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    print(f"Training {model_type.upper()} model...")
    if model_type.lower() == 'svm':
        pipeline = train_svm(X_train, y_train)
    else:
        pipeline = train_naive_bayes(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(pipeline, X_test, y_test)

    save_model(pipeline, model_path)


if __name__ == "__main__":
    # Demo training
    print("Training module - use main.py to train model")
