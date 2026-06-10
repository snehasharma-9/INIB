"""
03_feature_engineering.py - Extract TF-IDF, embeddings, and sentiment features
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Loading processed dataset...")
df = pd.read_csv('fake_news_processed.csv')

print(f"Dataset shape: {df.shape}")

# ============================================================================
# Feature 1: TF-IDF Vectorization
# ============================================================================
print("\n" + "="*80)
print("FEATURE 1: TF-IDF VECTORIZATION")
print("="*80)

tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,
    min_df=5,
    max_df=0.75,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r'(?u)\b\w\w+\b'
)

print("Fitting TF-IDF vectorizer (this may take a moment)...")
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])
print(f"TF-IDF shape: {tfidf_features.shape}")

# Save vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("OK TF-IDF vectorizer saved")

# ============================================================================
# Feature 2: Sentiment Analysis
# ============================================================================
print("\n" + "="*80)
print("FEATURE 2: SENTIMENT ANALYSIS")
print("="*80)

print("Extracting sentiment features...")
sentiments = []
for idx, text in enumerate(df['processed_text']):
    if idx % 2000 == 0:
        print(f"  Processing {idx}/{len(df)}...")
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
    except:
        sentiment = 0.0
    sentiments.append(sentiment)

df['sentiment'] = sentiments
print(f"Sentiment range: [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")
print(f"Average sentiment: {df['sentiment'].mean():.3f}")

# ============================================================================
# Feature 3: Text Statistics
# ============================================================================
print("\n" + "="*80)
print("FEATURE 3: TEXT STATISTICS")
print("="*80)

df['avg_word_length'] = df['processed_text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
)

df['unique_word_ratio'] = df['processed_text'].apply(
    lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
)

print(f"Average word length: {df['avg_word_length'].mean():.2f}")
print(f"Unique word ratio: {df['unique_word_ratio'].mean():.2f}")

# ============================================================================
# Feature 4: Bag of Words
# ============================================================================
print("\n" + "="*80)
print("FEATURE 4: BAG OF WORDS (CountVectorizer)")
print("="*80)

bow_vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.8, token_pattern=r'(?u)\b\w\w+\b')
print("Fitting Bag of Words vectorizer...")
bow_features = bow_vectorizer.fit_transform(df['processed_text'])
print(f"BoW shape: {bow_features.shape}")

# Save BoW vectorizer
with open('bow_vectorizer.pkl', 'wb') as f:
    pickle.dump(bow_vectorizer, f)
print("OK BoW vectorizer saved")

# ============================================================================
# Save Features and Metadata
# ============================================================================
print("\n" + "="*80)
print("SAVING FEATURES")
print("="*80)

# Create feature dataframe
feature_df = df[['label', 'text_length', 'word_count', 'sentiment', 'avg_word_length', 'unique_word_ratio']].copy()
feature_df['label_encoded'] = (feature_df['label'] == 'fake').astype(int)

# Save statistical features
feature_df.to_csv('statistical_features.csv', index=False)
print("OK Statistical features saved to 'statistical_features.csv'")

# Save TF-IDF features
np.save('tfidf_features.npy', tfidf_features.toarray())
print("OK TF-IDF features saved to 'tfidf_features.npy'")

# Save BoW features
np.save('bow_features.npy', bow_features.toarray())
print("OK BoW features saved to 'bow_features.npy'")

# ============================================================================
# Feature Summary
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"Statistical features: {feature_df.shape[1] - 1}")
print(f"TF-IDF features: {tfidf_features.shape[1]}")
print(f"BoW features: {bow_features.shape[1]}")
print(f"Total samples: {len(df)}")
print(f"\nFeature files created:")
print(f"  - tfidf_vectorizer.pkl")
print(f"  - bow_vectorizer.pkl")
print(f"  - statistical_features.csv")
print(f"  - tfidf_features.npy")
print(f"  - bow_features.npy")

print(f"\nFeature engineering complete!")
