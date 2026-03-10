from src.config import *
from src.models import *
from src.data import TextDataset, build_vocab, make_weighted_sampler
from src.train import train_model
from src.utils import set_seed, save_experiment
from src.metrics import compute_class_distribution

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
    "hidden_dim": HIDDEN_DIM,
    "embed_dim": EMBED_DIM,
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

print("Training class distribution:", compute_class_distribution(train_labels.values))
print("Validation class distribution:", compute_class_distribution(val_labels.values))

vocab = build_vocab(train_texts)

train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, vocab, MAX_LEN)
sampler = make_weighted_sampler(train_labels.values)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, criterion, optimizer
print("Initializing model, criterion, and optimizer...")
model = EmbeddingAttentionClassifier(len(vocab), EMBED_DIM).to(DEVICE)
# model = (len(vocab), EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,weight_decay=1e-5)

# Train
print("Training on the dataset...")
train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS)

# Save checkpoint
entry_dir = save_experiment(
    model=model,
    optimizer=optimizer,
    vocab=vocab,
    seed=SEED,
    max_len=MAX_LEN,
    config=config
)