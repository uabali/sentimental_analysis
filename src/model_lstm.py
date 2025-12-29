"""
LSTM Model for Sentiment Analysis
=================================
Basic LSTM model for 3-class sentiment classification.
This serves as baseline model for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    Basic LSTM model for sentiment classification.
    
    Architecture:
        Embedding -> LSTM -> Dropout -> Fully Connected -> Softmax
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        output_dim: Number of output classes (3 for sentiment)
        n_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        pad_idx: Index of padding token
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        dropout=0.5,
        bidirectional=False,
        pad_idx=0
    ):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Store config for saving/loading
        self.config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'pad_idx': pad_idx
        }
        
    def forward(self, input_ids, lengths=None):
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            lengths: Tensor of actual sequence lengths (optional)
            
        Returns:
            logits: Tensor of shape (batch_size, output_dim)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_dim)
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout and fully connected
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits
    
    def predict(self, input_ids, lengths=None):
        """Get predicted class."""
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, input_ids, lengths=None):
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(input_ids, lengths)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class BiLSTMModel(LSTMModel):
    """
    Bidirectional LSTM model.
    Simply sets bidirectional=True in parent class.
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        dropout=0.5,
        pad_idx=0
    ):
        super(BiLSTMModel, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            pad_idx=pad_idx
        )


# ==================== TESTING ====================

if __name__ == "__main__":
    # Test model
    vocab_size = 10000
    batch_size = 32
    seq_length = 50
    
    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths = torch.randint(10, seq_length, (batch_size,))
    
    # Test LSTM model
    print("Testing LSTM Model:")
    model = LSTMModel(vocab_size=vocab_size)
    output = model(input_ids, lengths)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test BiLSTM model
    print("\nTesting BiLSTM Model:")
    model_bi = BiLSTMModel(vocab_size=vocab_size)
    output_bi = model_bi(input_ids, lengths)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output_bi.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_bi.parameters()):,}")
