# %%
%pip install torch

# %%
import pandas as pd
import re


df = pd.read_csv("train_data.csv")
print(df.head())

# %%
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.split()

#config to grab words that appear more than 2 times

word_to_idx = {"<PAD>": 0, "<UNK>": 1}

idx = word_to_idx.get(token, word_to_idx["<UNK>"])

if len(seq) < max_len:
    seq += [0] * (max_len - len(seq))
else:
    seq = seq[:max_len]

# %%
from torch.utils.data import Dataset

class CommonsenseDataset(Dataset):
    def __init__(self, df, vocab):
        ...
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return input_tensor, label

# %%



