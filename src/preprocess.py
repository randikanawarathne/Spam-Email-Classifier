"""Text preprocessing module for spam email classification."""

import re
import string
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize email text.

    Args:
        text: Raw email text

    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words.

    Args:
        text: Cleaned text

    Returns:
        List of tokens
    """
    return text.split()


def preprocess_email(text: str) -> str:
    """Full preprocessing pipeline for email text.

    Args:
        text: Raw email text

    Returns:
        Preprocessed text ready for vectorization
    """
    return clean_text(text)


if __name__ == "__main__":
    # Test preprocessing
    sample = "URGENT! You've won $1,000,000! Click here: http://spam.com"
    print(f"Original: {sample}")
    print(f"Processed: {preprocess_email(sample)}")
