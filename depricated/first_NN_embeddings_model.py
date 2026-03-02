# %%
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

# %%
# MAX LEN IS PICKED BY LOOKING AT THE MEAN TOKEN LENGTH IN sentence_length.ipynb
MAX_LEN = 25

df = pd.read_csv("train_binary.csv")

X_train = df["input_text"]

# %%
# will implement spaCy later
def tokenizer(text):
    return text.lower().split()

# %%
counter = Counter()

for text in X_train:
    counter.update(tokenizer(text))

vocab = {word: i+1 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0

# %%
def encode(text, vocab):
    tokens = tokenizer(text)
    return [vocab.get(token, 0) for token in tokens]

# %%
def pad_sequence(seq, max_len=MAX_LEN):
    seq = seq[:max_len]
    return seq + [0] * (max_len - len(seq))

# %%
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

# %% [markdown]
# # EMBEDING CLASSIFIER

# %%
import sys
print(sys.executable)

import torch
print(torch.__file__)
print(torch.__version__)

import torch
print(hasattr(torch, "_utils"))

# %%
class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        
        embeds = self.embedding(x)

        pooled = embeds.mean(dim=1)

        logits = self.classifier(pooled)

        return logits.squeeze(1)

# %%
model = EmbeddingClassifier(len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(len(train_dataset))
print(train_dataset[0])

train_dataset = TextDataset(
    texts=df["input_text"],
    labels=df["label"],
    vocab=vocab
)

# %%
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TextDataset(df["input_text"], df["label"], vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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

# %%
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Train Accuracy:", correct / total)

# %%
#TODO ADD IN HANDCRAFTED FEATURES FOR BETTER MODEL ACCURACY


