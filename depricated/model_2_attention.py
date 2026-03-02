import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sys
import os, random
import numpy as np
import torch

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



set_seed(SEED)
# MAX LEN IS PICKED BY LOOKING AT THE MEAN TOKEN LENGTH IN sentence_length.ipynb
MAX_LEN = 25

df = pd.read_csv("train_binary.csv")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["input_text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
# will implement spaCy later
def tokenizer(text):
    return text.lower().split()
counter = Counter()

for text in train_texts:
    counter.update(tokenizer(text))

vocab = {"<PAD>": 0, "<UNK>": 1}
for i, (word, _) in enumerate(counter.most_common(10000), start=2):
    vocab[word] = i
    
def encode(text, vocab):
    tokens = tokenizer(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_len=MAX_LEN):
    seq = seq[:max_len]
    return seq + [0] * (max_len - len(seq))

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.data = [
            pad_sequence(encode(text, vocab), MAX_LEN)
            for text in texts
        ]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            self.labels[idx]
        )
    
train_dataset = TextDataset(train_texts, train_labels, vocab)
val_dataset   = TextDataset(val_texts,   val_labels,   vocab)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    generator=g,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    generator=g,
    num_workers=0
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda version:" + torch.version.cuda if torch.cuda.is_available() else "cuda not available")
print("SEED:", SEED)
print("MAX_LEN:", MAX_LEN)
print("Vocab size:", len(vocab))
print("Train size:", len(train_dataset), "Val size:", len(val_dataset))
print("Device:", device)
# EMBEDING CLASSIFIER
print(sys.executable)
print(torch.__file__)
print(torch.__version__)
print(hasattr(torch, "_utils"))
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        # x: [B, T, H]
        # mask: [B, T]

        scores = self.attn(x).squeeze(-1)    # [B, T]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)  # [B, T]

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return pooled

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )
        
        self.attention_pooling = AttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )



    def forward(self, x):

        # x: [B, T]
        
        embeds = self.embedding(x) # x: [B,T,D]

        # FIRST UPDATE: FIX MASK
        # mask = (x != 0).unsqueeze(-1)          # [B, T, 1]
        # embeds = embeds * mask                 # zero-out PAD embeddings
        # lengths = mask.sum(dim=1).clamp(min=1) # [B, 1]

        # pooled = embeds.sum(dim=1) / lengths   # [B, D]

        # SECOND UPDATE: ADD ATTENTION
        pooled = self.attention_pooling(embeds, x != 0)  # [B, D]

        logits = self.classifier(pooled)

        return logits.squeeze(1)


model = EmbeddingClassifier(len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model = EmbeddingClassifier(len(vocab)).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # logits
        preds = (torch.sigmoid(outputs) > 0.5)
        labels_bool = (labels > 0.5)

        correct += (preds == labels_bool).sum().item()
        total += labels.size(0)

print("Val Accuracy:", correct / total)
# checkpointing
import json, os

os.makedirs("artifacts", exist_ok=True)

torch.save(
    {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "seed": SEED,
        "max_len": MAX_LEN,
        "vocab": vocab,
    },
    "artifacts/checkpoint.pt"
)

with open("artifacts/run_config.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "max_len": MAX_LEN,
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": EPOCHS,
            "torch_version": torch.__version__,
        },
        f,
        indent=2,
    )
#TODO ADD IN HANDCRAFTED FEATURES FOR BETTER MODEL ACCURACY

#ADD CHECKPOINTING SO WE CAN REUSE A TRAINED MODEL ON THE ACTUAL TEST SET