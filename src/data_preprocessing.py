"""
Data Preprocessing Module for YouTube Comments Sentiment Analysis
=================================================================
This module handles all data preprocessing tasks including:
- Text cleaning and normalization
- Tokenization and vocabulary building
- Data splitting and batching
"""

import re
import string
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== TEXT CLEANING ====================

def clean_text(text):
    """
    Clean and normalize text data.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions (@username)
    4. Remove hashtags
    5. Remove HTML tags
    6. Remove emojis (keeping only ASCII)
    7. Remove extra whitespace
    8. Remove punctuation
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


# ==================== VOCABULARY ====================

class Vocabulary:
    """
    Vocabulary class for text tokenization.
    
    Attributes:
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
        word_freq: Counter for word frequencies
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq=2, max_vocab_size=50000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
    def build_vocabulary(self, texts):
        """Build vocabulary from list of texts."""
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Filter by minimum frequency and max vocab size
        filtered_words = [
            word for word, freq in self.word_freq.most_common(self.max_vocab_size)
            if freq >= self.min_freq
        ]
        
        # Add special tokens
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.UNK_TOKEN] = 1
        
        # Add vocabulary words
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary built: {len(self.word2idx)} words")
        return self
    
    def encode(self, text):
        """Convert text to list of indices."""
        words = text.split()
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices):
        """Convert list of indices back to text."""
        return ' '.join([self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size
            }, f)
        print(f"Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(min_freq=data['min_freq'], max_vocab_size=data['max_vocab_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']
        print(f"Vocabulary loaded from {path}: {len(vocab)} words")
        return vocab


# ==================== DATASET ====================

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Analysis.
    
    Args:
        texts: List of text samples
        labels: List of sentiment labels (0, 1, 2)
        vocab: Vocabulary object
        max_length: Maximum sequence length
    """
    
    LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
    LABEL_NAMES = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text)
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            encoded = encoded + [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(min(len(self.vocab.encode(self.texts[idx])), self.max_length), dtype=torch.long)
        }


# ==================== DATA LOADING ====================

def load_and_preprocess_data(
    data_path,
    text_column='CommentText',
    label_column='Sentiment',
    sample_size=None,
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        sample_size: Number of samples to use (None for all)
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Dictionary containing train/val/test data and vocabulary
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sample data if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
        print(f"Sampled {len(df)} rows")
    
    # Remove null values
    df = df.dropna(subset=[text_column, label_column])
    print(f"After removing nulls: {len(df)} rows")
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"After removing empty texts: {len(df)} rows")
    
    # Convert labels to lowercase
    df[label_column] = df[label_column].str.lower()
    
    # Map labels to integers
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df[label_column].map(label_map)
    
    # Print label distribution
    print("\nLabel Distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Split data
    texts = df['cleaned_text'].tolist()
    labels = df['label'].tolist()
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Build vocabulary from training data only
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=2, max_vocab_size=50000)
    vocab.build_vocabulary(X_train)
    
    return {
        'train': {'texts': X_train, 'labels': y_train},
        'val': {'texts': X_val, 'labels': y_val},
        'test': {'texts': X_test, 'labels': y_test},
        'vocab': vocab,
        'label_map': label_map
    }


def create_data_loaders(data_dict, vocab, batch_size=64, max_length=128, num_workers=0):
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        vocab: Vocabulary object
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        
    Returns:
        Dictionary of DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = SentimentDataset(
            texts=data_dict[split]['texts'],
            labels=data_dict[split]['labels'],
            vocab=vocab,
            max_length=max_length
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"{split.capitalize()} DataLoader created: {len(loaders[split])} batches")
    
    return loaders


# ==================== MAIN ====================

if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_path = Path(__file__).parent.parent / "data" / "youtube_comments_cleaned.csv"
    
    # Load data (using small sample for testing)
    data = load_and_preprocess_data(data_path, sample_size=10000)
    
    # Create data loaders
    loaders = create_data_loaders(data, data['vocab'], batch_size=32)
    
    # Test a batch
    batch = next(iter(loaders['train']))
    print(f"\nSample batch:")
    print(f"  Input shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Lengths shape: {batch['length'].shape}")
