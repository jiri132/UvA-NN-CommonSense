from src.config import *
from src.models import *
from src.data import TextDataset, build_vocab
from src.train import train_model
from src.utils import set_seed, save_experiment
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

config = {
    "model_name": MODEL_NAME,
    "seed": SEED,
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCHS
}

set_seed(SEED)

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["input_text"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
)
vocab = build_vocab(train_texts)

train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, criterion, optimizer
print("Initializing model, criterion, and optimizer...")
model = BiLSTMAttentionClassifier(len(vocab), EMBED_DIM, hidden_dim=64).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,weight_decay=1e-5)


# Save checkpoint
entry_dir = save_experiment(
    model=model,
    optimizer=optimizer,
    vocab=vocab,
    seed=SEED,
    max_len=MAX_LEN,
    config=config
)

# Train
print("Training on the dataset...")
train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS)