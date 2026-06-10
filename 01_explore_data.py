"""
01_explore_data.py - Initial data exploration and analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load dataset
df = pd.read_csv('fake_news_dataset.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)
print(df.info())

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Count'] > 0])

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)
class_dist = df['label'].value_counts()
print(class_dist)
print(f"\nClass Balance: {class_dist['fake']/len(df)*100:.1f}% fake, {class_dist['real']/len(df)*100:.1f}% real")

print("\n" + "=" * 80)
print("TEXT STATISTICS")
print("=" * 80)
df['title_length'] = df['title'].str.len()
df['text_length'] = df['text'].str.len()
df['title_words'] = df['title'].str.split().str.len()
df['text_words'] = df['text'].str.split().str.len()

print(df[['title_length', 'text_length', 'title_words', 'text_words']].describe())

print("\n" + "=" * 80)
print("CATEGORY DISTRIBUTION")
print("=" * 80)
print(df['category'].value_counts())

print("\n" + "=" * 80)
print("SOURCE DISTRIBUTION (Top 10)")
print("=" * 80)
print(df['source'].value_counts().head(10))

# Save statistics
stats = {
    'total_samples': len(df),
    'fake_samples': (df['label'] == 'fake').sum(),
    'real_samples': (df['label'] == 'real').sum(),
    'missing_source': df['source'].isnull().sum(),
    'missing_author': df['author'].isnull().sum(),
    'avg_text_length': df['text_length'].mean(),
    'avg_text_words': df['text_words'].mean(),
}

print("\n" + "=" * 80)
print("KEY STATISTICS")
print("=" * 80)
for key, value in stats.items():
    print(f"{key}: {value:.0f}" if isinstance(value, float) else f"{key}: {value}")

print("\nExploration complete! Dataset is ready for preprocessing.")
