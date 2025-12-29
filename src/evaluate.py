"""
Model Evaluation Module
=======================
Comprehensive evaluation with metrics and visualizations.
"""

import sys
from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_and_preprocess_data, create_data_loaders, Vocabulary
from src.model_lstm import LSTMModel, BiLSTMModel
from src.model_bilstm_attention import BiLSTMAttention


def load_model(model_path, ModelClass, vocab_size, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config'].copy()
    
    # Remove 'bidirectional' key if present - BiLSTMModel sets this internally
    config.pop('bidirectional', None)
    
    model = ModelClass(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)
            
            outputs = model(input_ids, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    class_names = ['Negative', 'Neutral', 'Positive']
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': {
            class_names[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': int(support[i])
            }
            for i in range(3)
        }
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def plot_training_history(history_path, save_path=None):
    """Plot training history."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def compare_models(results_dict, save_path=None):
    """Compare multiple models."""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = {metric: [results_dict[m][metric] for m in models] for metric in metrics}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig


def main():
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "data" / "youtube_comments_cleaned.csv"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\nLoading data...")
    data = load_and_preprocess_data(DATA_PATH, sample_size=100000, test_size=0.2, val_size=0.1)
    loaders = create_data_loaders(data, data['vocab'], batch_size=64, max_length=128)
    
    # Models to evaluate
    models_info = [
        ("LSTM", LSTMModel),
        ("BiLSTM", BiLSTMModel),
        ("BiLSTM_Attention", BiLSTMAttention),
    ]
    
    all_results = {}
    
    for model_name, ModelClass in models_info:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print("="*60)
        
        model_path = MODELS_DIR / f"{model_name}_best.pt"
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        # Load model
        model, checkpoint = load_model(model_path, ModelClass, len(data['vocab']), DEVICE)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        y_pred, y_true, y_proba = evaluate_model(model, loaders['test'], DEVICE)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        all_results[model_name] = metrics
        
        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # Plot training history
        history_path = MODELS_DIR / f"{model_name}_history.json"
        if history_path.exists():
            hist_path = RESULTS_DIR / f"{model_name}_training_history.png"
            plot_training_history(history_path, hist_path)
        
        # Save metrics
        metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    # Compare models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Model Comparison")
        print("="*60)
        comparison_path = RESULTS_DIR / "model_comparison.png"
        compare_models(all_results, comparison_path)
        
        # Print comparison table
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*68)
        for model_name, metrics in all_results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
