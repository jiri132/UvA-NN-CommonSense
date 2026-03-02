import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset

def tokenizer(text: str):
    return text.lower().split()

def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab_size), start=2):
        vocab[word] = i
    return vocab

def encode(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenizer(text)]

def pad_sequence(seq, max_len):
    seq = seq[:max_len]
    return seq + [0] * (max_len - len(seq))

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.data = [pad_sequence(encode(text, vocab), max_len) for text in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), self.labels[idx]