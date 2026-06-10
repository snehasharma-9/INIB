"""
02_preprocess_data.py - Data cleaning and preprocessing
"""
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data quietly
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

print("Loading dataset...")
df = pd.read_csv('fake_news_dataset.csv')

# Handle missing values
print(f"\nHandling missing values...")
df['source'] = df['source'].fillna('Unknown')
df['author'] = df['author'].fillna('Unknown')

# Create combined text feature
df['combined_text'] = df['title'] + ' ' + df['text']

# Text cleaning function
def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
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
    
    return text

print("\nCleaning text...")
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Remove stopwords and apply stemming
print("Removing stopwords and stemming...")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Tokenize, remove stopwords, and stem"""
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

# Remove empty texts
print(f"Samples before filtering: {len(df)}")
df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
print(f"Samples after filtering: {len(df)}")

# Add text length features
df['text_length'] = df['processed_text'].str.len()
df['word_count'] = df['processed_text'].str.split().str.len()

print(f"\nText statistics after preprocessing:")
print(f"Average text length: {df['text_length'].mean():.0f}")
print(f"Average word count: {df['word_count'].mean():.0f}")

# Save processed dataset
output_file = 'fake_news_processed.csv'
df_save = df[['processed_text', 'label', 'text_length', 'word_count', 'category', 'source']].copy()
df_save.to_csv(output_file, index=False)
print(f"\nProcessed dataset saved to '{output_file}'")

# Display sample
print(f"\n--- Sample Processed Data ---")
print(f"Original: {df['combined_text'].iloc[0][:100]}...")
print(f"Processed: {df['processed_text'].iloc[0][:100]}...")
print(f"Label: {df['label'].iloc[0]}")

print(f"\nPreprocessing complete!")
print(f"Output file: {output_file}")
