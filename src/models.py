import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

class EmbeddingMeanClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention_pooling = AttentionPooling(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embeds = self.embedding(x)             # x: [B,T,D]
        mask = (x != 0).unsqueeze(-1)          # [B, T, 1]
        embeds = embeds * mask                 # zero-out PAD embeddings
        lengths = mask.sum(dim=1).clamp(min=1) # [B, 1]
        pooled = embeds.sum(dim=1) / lengths   # [B, D]

        logits = self.classifier(pooled)
        return logits.squeeze(1)

class EmbeddingAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention_pooling = AttentionPooling(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mask = x != 0
        embeds = self.embedding(x)
        pooled = self.attention_pooling(embeds, mask)
        logits = self.classifier(pooled)
        return logits.squeeze(1)

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = AttentionPooling(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mask = x != 0
        embeds = self.embedding(x) 
        lstm_out, _ = self.bilstm(embeds) 
        pooled = self.attention(lstm_out, mask)
        logits = self.classifier(pooled)
        return logits.squeeze(1)

