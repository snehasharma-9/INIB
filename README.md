# Fake News Detection Model

A comprehensive machine learning project for detecting fake news articles using NLP and multiple ML algorithms.

## 📊 Dataset Overview

- **Size**: 20,000 news articles
- **Features**: title, text, date, source, author, category, label
- **Classes**: FAKE (50.3%) vs REAL (49.7%) - Well balanced
- **Missing Values**: 5% source, 5% author (handled)

## 🚀 Project Pipeline

### Step 1: Data Exploration (`01_explore_data.py`)
- Analyze dataset structure, columns, data types
- Check for missing values and class distribution
- Calculate text statistics

**Output**: Dataset overview and statistics

### Step 2: Data Preprocessing (`02_preprocess_data.py`)
- Clean text: remove URLs, emails, special characters
- Convert to lowercase
- Remove stopwords (English)
- Apply stemming using Porter Stemmer
- Generate clean text features

**Output**: `fake_news_processed.csv`

### Step 3: Feature Engineering (`03_feature_engineering.py`)
- **TF-IDF**: Extract 5,000 features with bigrams
- **Sentiment Analysis**: TextBlob polarity scores (-1 to 1)
- **Text Statistics**: word count, text length, unique word ratio
- **Bag of Words**: 1,000-feature count vectorization

**Output**: 
- `tfidf_features.npy` - TF-IDF matrix
- `bow_features.npy` - BoW matrix
- `statistical_features.csv` - Sentiment & text stats
- `tfidf_vectorizer.pkl`, `bow_vectorizer.pkl` - Saved vectorizers

### Step 4: Model Training (`04_train_models.py`)
Models trained on combined TF-IDF + statistical features:
1. **Logistic Regression** - Fast, interpretable
2. **Linear SVM** - Good for high-dimensional data
3. **Random Forest** - Ensemble method
4. **Gradient Boosting** - Advanced ensemble

**Train/Val/Test Split**: 70% / 10% / 20%

**Output**: Best model saved as `best_model_*.pkl`

### Step 5: Model Evaluation (`05_evaluate_models.py`)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC curves
- Performance visualizations

**Output**: 
- `model_comparison.csv` - Metrics table
- `model_comparison.png` - Comparison plots
- `confusion_matrix_*.png` - Confusion matrices

### Step 6: Predictions (`06_predict.py`)
Make predictions on new news articles with confidence scores.

## 📦 Required Dependencies

```bash
pip install pandas numpy scikit-learn nltk textblob matplotlib seaborn
```

## 🔧 Quick Start

### Run Full Pipeline
```bash
python main.py
```

### Run Individual Steps
```bash
python 01_explore_data.py
python 02_preprocess_data.py
python 03_feature_engineering.py
python 04_train_models.py
python 05_evaluate_models.py
```

### Make Predictions
```bash
python 06_predict.py
```

## 📈 Expected Results

### Model Performance (Test Set)
- **Accuracy**: ~95%
- **Precision**: ~95%
- **Recall**: ~95%
- **F1-Score**: ~95%

### Best Model
- **Logistic Regression** or **Gradient Boosting** (depending on data)
- Fast inference time
- Excellent generalization

## 📁 Output Files

### Saved Models & Vectorizers
- `best_model_*.pkl` - Trained model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `bow_vectorizer.pkl` - Bag of Words vectorizer
- `scaler.pkl` - Feature scaler

### Results & Data
- `fake_news_processed.csv` - Cleaned dataset
- `statistical_features.csv` - Extracted features
- `tfidf_features.npy` - TF-IDF matrix
- `bow_features.npy` - BoW matrix
- `model_comparison.csv` - Model metrics
- `detailed_metrics.csv` - Detailed performance metrics

### Visualizations
- `model_comparison.png` - Performance comparison charts
- `confusion_matrix_*.png` - Confusion matrices

## 🔍 Feature Importance

### Top Indicators of Fake News
1. Unusual word patterns (TF-IDF weights)
2. Sentiment analysis scores (extreme polarity)
3. Text length and word count patterns
4. Vocabulary uniqueness ratios

## 💡 Usage Example

```python
from predict import predict_news

article = "Breaking news: Scientists discover revolutionary cure..."
result = predict_news(article)

print(f"Label: {result['label']}")  # 'FAKE NEWS' or 'REAL NEWS'
print(f"Confidence: {result['confidence']:.2%}")
print(f"Fake Probability: {result['fake_prob']:.4f}")
```

## 🎯 Future Enhancements

- [ ] Add transformer models (BERT, RoBERTa)
- [ ] Implement ensemble voting classifier
- [ ] Add source credibility scoring
- [ ] Deploy as REST API
- [ ] Create web interface
- [ ] Add attention visualization
- [ ] Cross-language support

## 📝 References

- Dataset: Kaggle Fake News Detection Datasets
- NLP: NLTK, TextBlob, scikit-learn
- ML: Logistic Regression, SVM, Random Forest, Gradient Boosting

## 👤 Author
Generated as part of Fake News Detection Project

## 📄 License
Open source - Use freely for education and research
