"""
05_evaluate_models.py - Comprehensive evaluation and visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import pickle

sns.set_style("whitegrid")

print("="*80)
print("LOADING RESULTS")
print("="*80)

# Load results
with open('model_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Load test data
stat_features = pd.read_csv('statistical_features.csv')
tfidf_features = np.load('tfidf_features.npy')
y_all = stat_features['label_encoded'].values

print(f"Loaded results for {len(results)} models")

# ============================================================================
# Detailed Metrics
# ============================================================================
print("\n" + "="*80)
print("DETAILED METRICS FOR ALL MODELS")
print("="*80)

metrics_list = []
for model_name in results.keys():
    metrics_list.append({
        'Model': model_name,
        'Train Accuracy': results[model_name]['train_acc'],
        'Val Accuracy': results[model_name]['val_acc'],
        'Test Accuracy': results[model_name]['test_acc'],
        'Precision': results[model_name]['precision'],
        'Recall': results[model_name]['recall'],
        'F1-Score': results[model_name]['f1']
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.to_string(index=False))

# Save detailed metrics
metrics_df.to_csv('detailed_metrics.csv', index=False)
print("\nOK Metrics saved to 'detailed_metrics.csv'")

# ============================================================================
# Model Performance Visualization
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax = axes[0, 0]
models = metrics_df['Model'].values
train_acc = metrics_df['Train Accuracy'].values
val_acc = metrics_df['Val Accuracy'].values
test_acc = metrics_df['Test Accuracy'].values

x = np.arange(len(models))
width = 0.25

ax.bar(x - width, train_acc, width, label='Train', alpha=0.8)
ax.bar(x, val_acc, width, label='Val', alpha=0.8)
ax.bar(x + width, test_acc, width, label='Test', alpha=0.8)

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: F1-Score Comparison
ax = axes[0, 1]
f1_scores = metrics_df['F1-Score'].values
colors = ['#2ecc71' if score == f1_scores.max() else '#3498db' for score in f1_scores]
ax.barh(models, f1_scores, color=colors, alpha=0.8)
ax.set_xlabel('F1-Score')
ax.set_title('F1-Score Comparison (Best: Green)')
ax.set_xlim([0, 1])
for i, v in enumerate(f1_scores):
    ax.text(v - 0.05, i, f'{v:.4f}', va='center', ha='right', fontweight='bold', color='white')

# Plot 3: Precision vs Recall
ax = axes[1, 0]
precision = metrics_df['Precision'].values
recall = metrics_df['Recall'].values
for i, model in enumerate(models):
    ax.scatter(recall[i], precision[i], s=200, alpha=0.6, label=model)
    ax.annotate(model, (recall[i], precision[i]), fontsize=8, ha='right')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(alpha=0.3)

# Plot 4: Overall Performance Heatmap
ax = axes[1, 1]
heatmap_data = metrics_df[['Precision', 'Recall', 'F1-Score', 'Test Accuracy']].values
im = ax.imshow(heatmap_data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'Accuracy'])
ax.set_yticklabels(models)
ax.set_title('Performance Heatmap')

# Add text annotations
for i in range(len(models)):
    for j in range(4):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("OK Model comparison plot saved to 'model_comparison.png'")
plt.close()

# ============================================================================
# Best Model Details
# ============================================================================
print("\n" + "="*80)
print("BEST MODEL ANALYSIS")
print("="*80)

best_idx = metrics_df['F1-Score'].idxmax()
best_model_name = metrics_df.loc[best_idx, 'Model']
print(f"Best Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"Accuracy: {results[best_model_name]['test_acc']:.4f}")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_all[-len(results[best_model_name]['predictions']):], 
                     results[best_model_name]['predictions'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{best_model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
print(f"OK Confusion matrix plot saved")
plt.close()

print(f"\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"All metrics and visualizations have been saved!")
