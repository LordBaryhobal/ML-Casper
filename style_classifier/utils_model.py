# src/utils_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import polars as pl
import math


class ChessGameDataset(Dataset):
    """
    Loads the preprocessed Parquet file and provides
    (token_ids, label_id) pairs as Tensors.
    """
    def __init__(self, parquet_file):
        df = pl.read_parquet(parquet_file)
        
        # Convert to NumPy for faster indexing in __getitem__
        # .to_list() is faster for nested lists than .to_numpy(structured=True)
        self.token_ids = df['token_ids'].to_list() 
        self.labels = df['label_id'].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token_ids = self.token_ids[idx]
        label = self.labels[idx]
        
        # Convert to Tensors
        # .long() is equivalent to torch.int64
        return torch.tensor(token_ids, dtype=torch.long), \
               torch.tensor(label, dtype=torch.long)
    
    def print_shape_info(self):
        print(f"Dataset size: {len(self)}")
        sample_token_ids, sample_label = self[0]
        print(f"Sample token_ids shape: {sample_token_ids.shape}")
        print(f"Sample label shape: {sample_label.shape}")

# --- 5. Model Architecture ---

# --- Step 3: Positional Encoding --- parce que dans un transformer les entrées sont permutation invariantes (ce qui signifie si deux mots sont échangés, 
#  la sortie doit rester la même (ce qui est un problèmes pour une partie d'échecs 
#  ou on perd la notion de temporalité entre les coups)),
#  si on ne donne pas d'information de position au modèle il ne peut pas en apprendre.
class PositionalEncoding(nn.Module):
    # d_model: embedding dimension, on projette linéairement les tokens dans cet espace à d_model dimensions
    # le dropout est appliqué après l'ajout du positional encoding pour éviter l'overfitting - pas convaincu mais bon pour l'utilité
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        # Use batch_first dimensioning: [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # register_buffer makes 'pe' part of the module's state,
        # but not a trainable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input embeddings
        # self.pe is [1, max_len, d_model]
        # x is [batch_size, seq_len, d_model]
        # We take self.pe[:, :x.size(1)] to match the seq_len
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- Steps 2, 4, 5: The Full Transformer Model ---
class StyleTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float, seq_len: int):
        super().__init__()
        self.d_model = d_model

        # --- Step 2: Embedding Layer ---
        # padding_idx=0 tells the model to ignore [PAD] tokens (ID 0)
        # Embedding layer maps token IDs to dense vectors, c'est une lookup table qui stocke les représentations vectorielles, par défaut initialisées aléatoirement et apprises pendant l'entraînement.
        # vocab_size: taille du vocabulaire (nombre total de tokens uniques), comme il se peut que certains coups ne soient pas présents dans l'ensemble d'entraînement - à totaliser plus tard
        # je considère que le vocabulaire est complet et fixe après la préprocessing sur un dataset de taille suffisante
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # --- Step 3: Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # --- Step 4: Transformer Encoder Block ---
        # l'encoder_layer' est un bloc de base du Transformer qu'on empile pour former l'encodeur complet
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # This makes input (batch_size, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # --- Step 5: Classification Head ---
        # Takes the [CLS] token's output vector and maps it to class logits
        self.classification_head = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src shape: [batch_size, seq_len] e.g., (32, 20)
        """
        # print(src)
        # --- CRITICAL: Create Padding Mask ---
        # The model must ignore [PAD] tokens (ID 0) during self-attention
        # Shape: [batch_size, seq_len] e.g., (32, 20)
        # It's True where src == 0, False otherwise.
        src_key_padding_mask = (src == 0)

        # --- Step 2: Embedding ---
        # Shape: (32, 20) -> (32, 20, 256)
        x = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding
        
        # --- Step 3: Positional Encoding ---
        # Shape: (32, 20, 256) -> (32, 20, 256)
        x = self.pos_encoder(x)
        
        # --- Step 4: Transformer Encoder ---
        # Shape: (32, 20, 256) -> (32, 20, 256)
        # The mask ensures [PAD] tokens are ignored
        x = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask # cette fonction est souvent buggée - à tester soigneusement - ou faire à la main
        )
        
        # --- Step 5: Classification Head ---
        # Select the [CLS] token's output (at position 0)
        # Shape: (32, 20, 256) -> (32, 256)
        # x_cls = x[:, 0, :] # old, Problem: Your model (utils_model.py) relies exclusively on the [CLS] token's final representation (x = x[:, 0, :]). 
	# This forces the model to learn to compress all stylistic information into that single token.
	# Solution: Try Global Average Pooling as an alternative. This averages the representations of all move tokens, 
	# forcing every move in the (now much longer) sequence to contribute to the classification.
        x = x.mean(dim=1) # NEW: Global Average Pooling. Shape: [batch_size, d_model]
       
	# --- Step 6: Classification Head ---
	# Pass through the final linear layer
    # Shape: (32, 256) -> (32, num_classes)
        logits = self.classification_head(x)
    # logits = self.classification_head(x_cls)
        return logits
