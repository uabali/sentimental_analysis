"""
Training Module for Sentiment Analysis Models
==============================================
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_and_preprocess_data, create_data_loaders
from src.model_lstm import LSTMModel, BiLSTMModel
from src.model_bilstm_attention import BiLSTMAttention


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', learning_rate=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rate': []}
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, lengths)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / total, correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / total, correct / total
    
    def train(self, epochs, save_dir, model_name="model", early_stopping_patience=5):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        no_improvement_count = 0
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                no_improvement_count = 0
                
                model_path = save_dir / f"{model_name}_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.model.config
                }, model_path)
                print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            print()
        
        print(f"Best epoch: {self.best_epoch}, Best val acc: {self.best_val_acc:.4f}")
        
        history_path = save_dir / f"{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def main():
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "data" / "youtube_comments_cleaned.csv"
    SAVE_DIR = BASE_DIR / "models"
    
    SAMPLE_SIZE = 100000
    MAX_LENGTH = 128
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.5
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    data = load_and_preprocess_data(DATA_PATH, sample_size=SAMPLE_SIZE, test_size=0.2, val_size=0.1)
    loaders = create_data_loaders(data, data['vocab'], batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    
    vocab_path = SAVE_DIR / "vocabulary.pkl"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    data['vocab'].save(vocab_path)
    
    models_to_train = [
        ("LSTM", LSTMModel),
        ("BiLSTM", BiLSTMModel),
        ("BiLSTM_Attention", BiLSTMAttention),
    ]
    
    results = {}
    
    for model_name, ModelClass in models_to_train:
        model = ModelClass(
            vocab_size=len(data['vocab']),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=3,
            n_layers=N_LAYERS,
            dropout=DROPOUT
        )
        
        trainer = Trainer(model, loaders['train'], loaders['val'], DEVICE, LEARNING_RATE, WEIGHT_DECAY)
        history = trainer.train(EPOCHS, SAVE_DIR, model_name, EARLY_STOPPING_PATIENCE)
        
        results[model_name] = {
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'best_epoch': trainer.best_epoch
        }
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: Val Acc={result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
