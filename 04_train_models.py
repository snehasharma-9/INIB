"""
04_train_models.py - Train multiple ML models for fake news detection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOADING FEATURES")
print("="*80)

# Load features
stat_features = pd.read_csv('statistical_features.csv')
tfidf_features = np.load('tfidf_features.npy')
bow_features = np.load('bow_features.npy')

# Prepare labels
y = stat_features['label_encoded'].values

print(f"Statistical features shape: {stat_features.shape}")
print(f"TF-IDF features shape: {tfidf_features.shape}")
print(f"BoW features shape: {bow_features.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# Create combined feature set
# ============================================================================
print("\n" + "="*80)
print("CREATING COMBINED FEATURES")
print("="*80)

# Use only relevant statistical features
stat_cols = ['text_length', 'word_count', 'sentiment', 'avg_word_length', 'unique_word_ratio']
stat_subset = stat_features[stat_cols].values

# Combine TF-IDF with statistical features
X_combined = np.hstack([tfidf_features, stat_subset])
print(f"Combined features shape: {X_combined.shape}")

# Scale statistical features
scaler = StandardScaler()
stat_subset_scaled = scaler.fit_transform(stat_subset)
X_tfidf_scaled = np.hstack([tfidf_features, stat_subset_scaled])

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("OK Scaler saved")

# ============================================================================
# Train-Test Split
# ============================================================================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train/Val/Test ratio: {X_train.shape[0]/len(y):.1%} / {X_val.shape[0]/len(y):.1%} / {X_test.shape[0]/len(y):.1%}")

# ============================================================================
# Train Models
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42, dual=False),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    results[model_name] = {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'predictions': y_test_pred
    }
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {test_precision:.4f}")
    print(f"  Recall:         {test_recall:.4f}")
    print(f"  F1-Score:       {test_f1:.4f}")

# ============================================================================
# Model Comparison
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON (Test Set)")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['test_acc'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()]
})

print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
best_model = results[best_model_name]['model']
print(f"\nOK Best Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

# ============================================================================
# Save Best Model
# ============================================================================
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

model_file = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(best_model, f)
print(f"OK Best model saved to '{model_file}'")

# Save all results
with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("OK All model results saved")

# Save comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("OK Model comparison saved to 'model_comparison.csv'")

# ============================================================================
# Detailed Analysis of Best Model
# ============================================================================
print("\n" + "="*80)
print(f"DETAILED ANALYSIS: {best_model_name}")
print("="*80)

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, results[best_model_name]['predictions']))

print(f"\nClassification Report:")
print(classification_report(y_test, results[best_model_name]['predictions'], 
                          target_names=['Real News', 'Fake News']))

print(f"\nTraining complete!")
print(f"OK Best model: {best_model_name}")
print(f"OK Test F1-Score: {results[best_model_name]['f1']:.4f}")
