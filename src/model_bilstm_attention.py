"""
Bi-LSTM with Attention Model for Sentiment Analysis
====================================================
Advanced model combining Bidirectional LSTM with Attention mechanism.
This is the main model for the deep learning project.

Attention Mechanism:
- Allows the model to focus on important words
- Provides interpretability through attention weights
- Improves performance on longer sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention mechanism for sequence models.
    
    Computes attention weights for each timestep and returns
    a weighted sum of the hidden states.
    
    Args:
        hidden_dim: Dimension of hidden states
        attention_dim: Dimension of attention layer (optional)
    """
    
    def __init__(self, hidden_dim, attention_dim=None):
        super(Attention, self).__init__()
        
        attention_dim = attention_dim or hidden_dim
        
        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, lstm_output, mask=None):
        """
        Forward pass.
        
        Args:
            lstm_output: Tensor of shape (batch_size, seq_length, hidden_dim)
            mask: Optional mask for padding (batch_size, seq_length)
            
        Returns:
            context: Weighted sum of hidden states (batch_size, hidden_dim)
            attention_weights: Attention weights (batch_size, seq_length)
        """
        # Compute attention scores: (batch_size, seq_length, 1)
        attention_scores = self.attention(lstm_output)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_length)
        
        # Apply mask if provided (set padding positions to very negative value)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length)
        
        # Compute weighted sum (context vector)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_length)
            lstm_output  # (batch_size, seq_length, hidden_dim)
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention for sentiment classification.
    
    Architecture:
        Embedding -> Bi-LSTM -> Attention -> Dropout -> FC -> Softmax
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state (per direction)
        output_dim: Number of output classes
        n_layers: Number of LSTM layers
        dropout: Dropout probability
        pad_idx: Index of padding token
        attention_dim: Dimension of attention layer (optional)
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        dropout=0.5,
        pad_idx=0,
        attention_dim=128
    ):
        super(BiLSTMAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        # BiLSTM output dim is hidden_dim * 2
        self.attention = Attention(
            hidden_dim=hidden_dim * 2,
            attention_dim=attention_dim
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Store config for saving/loading
        self.config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'pad_idx': pad_idx,
            'attention_dim': attention_dim
        }
        
    def forward(self, input_ids, lengths=None, return_attention=False):
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            lengths: Tensor of actual sequence lengths (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Tensor of shape (batch_size, output_dim)
            attention_weights: Optional, attention weights (batch_size, seq_length)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create mask for attention (1 for real tokens, 0 for padding)
        mask = (input_ids != 0).float()  # (batch_size, seq_length)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths.cpu().clamp(min=1), 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, 
                batch_first=True, 
                total_length=seq_length
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention
        context, attention_weights = self.attention(lstm_out, mask)
        
        # Dropout and fully connected layers
        output = self.dropout(context)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.fc2(output)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict(self, input_ids, lengths=None):
        """Get predicted class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, input_ids, lengths=None):
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_attention_weights(self, input_ids, lengths=None):
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(input_ids, lengths, return_attention=True)
        return attention_weights


class CNNLSTMAttention(nn.Module):
    """
    Hybrid CNN-LSTM with Attention model.
    
    Combines CNN for local feature extraction with LSTM for sequential modeling.
    
    Architecture:
        Embedding -> CNN -> Bi-LSTM -> Attention -> FC -> Softmax
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_filters=100,
        filter_sizes=(3, 4, 5),
        n_layers=1,
        dropout=0.5,
        pad_idx=0
    ):
        super(CNNLSTMAttention, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=n_filters,
                kernel_size=fs,
                padding=fs // 2
            )
            for fs in filter_sizes
        ])
        
        # LSTM after CNN
        cnn_output_dim = n_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Attention
        self.attention = Attention(hidden_dim * 2)
        
        # Fully connected
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_filters': n_filters,
            'filter_sizes': filter_sizes,
            'n_layers': n_layers,
            'dropout': dropout,
            'pad_idx': pad_idx
        }
        
    def forward(self, input_ids, lengths=None, return_attention=False):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # CNN expects (batch, channels, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply CNN and concatenate
        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]
        cnn_out = torch.cat(conv_outputs, dim=1)  # (batch, n_filters * len(filter_sizes), seq_len)
        
        # Back to (batch, seq_len, features) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        
        # Attention
        mask = (input_ids != 0).float()
        context, attention_weights = self.attention(lstm_out, mask)
        
        # FC
        output = self.dropout(context)
        logits = self.fc(output)
        
        if return_attention:
            return logits, attention_weights
        return logits


# ==================== TESTING ====================

if __name__ == "__main__":
    # Test models
    vocab_size = 10000
    batch_size = 32
    seq_length = 50
    
    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths = torch.randint(10, seq_length, (batch_size,))
    
    # Test BiLSTM with Attention
    print("Testing BiLSTM + Attention Model:")
    model = BiLSTMAttention(vocab_size=vocab_size)
    output, attention = model(input_ids, lengths, return_attention=True)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test CNN-LSTM with Attention
    print("\nTesting CNN-LSTM + Attention Model:")
    model_cnn = CNNLSTMAttention(vocab_size=vocab_size)
    output_cnn, attention_cnn = model_cnn(input_ids, return_attention=True)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output_cnn.shape}")
    print(f"  Attention shape: {attention_cnn.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
