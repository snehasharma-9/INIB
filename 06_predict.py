"""
06_predict.py - Make predictions on new news articles
"""
import pickle
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved models and vectorizers
print("Loading model and vectorizers...")
with open('best_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords, stem
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def extract_features(text):
    """Extract all features for a news article"""
    processed_text = preprocess_text(text)
    
    # TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray()[0]
    
    # Statistical features
    text_length = len(processed_text)
    word_count = len(processed_text.split())
    sentiment = TextBlob(processed_text).sentiment.polarity
    avg_word_length = np.mean([len(word) for word in processed_text.split()]) if word_count > 0 else 0
    unique_word_ratio = len(set(processed_text.split())) / word_count if word_count > 0 else 0
    
    stat_features = np.array([text_length, word_count, sentiment, avg_word_length, unique_word_ratio])
    
    # Combine and scale
    combined_features = np.hstack([tfidf_features, stat_features])
    scaled_features = np.hstack([tfidf_features, scaler.transform([stat_features])[0]])
    
    return scaled_features


def align_features(features, model):
    expected = getattr(model, 'n_features_in_', None)
    if expected is None or features.shape[0] == expected:
        return features

    if features.shape[0] < expected:
        stat_length = 5
        if features.shape[0] <= stat_length:
            raise ValueError(f"Feature vector is too short: {features.shape[0]} features")
        tfidf_length = features.shape[0] - stat_length
        pad_length = expected - features.shape[0]
        print(f"Warning: padding feature vector from {features.shape[0]} to {expected} features")
        return np.hstack([features[:tfidf_length], np.zeros(pad_length), features[tfidf_length:]])

    raise ValueError(
        f"Feature mismatch: model expects {expected} features, got {features.shape[0]}"
    )


def predict_news(text):
    """Predict if news is fake or real"""
    features = extract_features(text)
    features = align_features(features, model)
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    label = 'FAKE NEWS' if prediction == 1 else 'REAL NEWS'
    confidence = probability[prediction]
    
    return {
        'label': label,
        'confidence': confidence,
        'real_prob': probability[0],
        'fake_prob': probability[1]
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*80)
    
    # Example news articles
    examples = [
        "Breaking: Scientists discover cure for all diseases using simple household items",
        "Local government approves new infrastructure project for community development",
        "Celebrity spotted at exclusive resort during vacation",
    ]
    
    for i, article in enumerate(examples, 1):
        print(f"\nArticle {i}:")
        print(f"Text: {article[:100]}...")
        
        result = predict_news(article)
        
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Real Probability: {result['real_prob']:.4f}")
        print(f"Fake Probability: {result['fake_prob']:.4f}")
