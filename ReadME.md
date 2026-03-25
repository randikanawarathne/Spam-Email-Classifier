# Spam Email Classifier - Project Specification

## 1. Project Overview

- **Project Name**: Spam Email Classifier
- **Type**: Machine Learning / NLP Application
- **Core Functionality**: A machine learning model that classifies emails as spam or ham (not spam) using text classification techniques
- **Target Users**: Anyone who wants to filter spam emails or learn about ML text classification

## 2. Technical Stack

- **Language**: Python 3.9+
- **ML Framework**: scikit-learn
- **Key Libraries**:
  - pandas (data handling)
  - numpy (numerical operations)
  - scikit-learn (ML models, feature extraction)
  - nltk (text preprocessing)
  - joblib (model serialization)

## 3. Features & Functionality

### Core Features
1. **Data Loading**: Load and parse email datasets (CSV format with label and text)
2. **Text Preprocessing**: Clean and normalize email text (lowercase, remove special chars, tokenization)
3. **Feature Extraction**: Convert text to numerical features using TF-IDF vectorization
4. **Model Training**: Train a Naive Bayes classifier on labeled email data
5. **Model Evaluation**: Evaluate model performance using accuracy, precision, recall, F1-score
6. **Spam Prediction**: Predict whether new emails are spam or ham
7. **Model Persistence**: Save trained model and vectorizer for later use

### Project Structure
```
spam-email-classifier/
├── data/
│   └── spam_dataset.csv        # Sample dataset
├── models/
│   └── spam_classifier.pkl     # Saved model
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Text preprocessing
│   ├── train.py                 # Model training script
│   ├── predict.py               # Prediction script
│   └── evaluate.py              # Evaluation script
├── main.py                      # Main CLI interface
├── requirements.txt
└── SPEC.md
```

## 4. Dataset Format

The dataset should be a CSV with columns:
- `text`: The email content
- `label`: "spam" or "ham"

## 5. Functionality Specification

### Text Preprocessing
- Convert to lowercase
- Remove special characters and punctuation
- Remove numbers (optional)
- Remove extra whitespace
- Tokenize text

### Model Training
- Split data: 80% training, 20% testing
- Use TF-IDF vectorization with:
  - Max features: 5000
  - N-grams: (1, 2) unigrams and bigrams
- Train Naive Bayes classifier (MultinomialNB)
- Support vector machine classifier (SVM) as alternative

### Prediction
- Accept raw email text
- Preprocess the text
- Transform using fitted vectorizer
- Predict using trained model
- Output: "SPAM" or "HAM" with confidence score

## 6. User Interface

### GUI Interface (main.py)
- **Framework**: Tkinter (built into Python)
- **Window Size**: 700x600 pixels
- **Components**:
  - Text input area (scrolled text widget)
  - "Classify Email" button to run prediction
  - "Train Model" button to train/re-train model
  - "Clear" button to reset input
  - "Load Sample" button to load test samples
  - Result display showing SPAM/HAM with confidence
  - Model status indicator

### GUI Layout
```
+------------------------------------------+
|         Spam Email Classifier            |
|  Model: Loaded [Train Model]             |
+------------------------------------------+
| Email Text                               |
| +--------------------------------------+ |
| | (Text input area)                    | |
| |                                      | |
| +--------------------------------------+ |
|                                          |
| [Classify Email] [Clear]    [Load Sample]|
+------------------------------------------+
| Classification Result                     |
| Result: SPAM                             |
| Confidence: 95.2%                        |
+------------------------------------------+
| About                                    |
| (Description text)                       |
+------------------------------------------+
```

## 7. Acceptance Criteria

1. Model trains successfully on spam/ham dataset
2. Model achieves >85% accuracy on test set
3. Predictions work for new email text
4. Model can be saved and loaded from disk
5. GUI interface launches and is functional
6. Code is clean, documented, and runnable
