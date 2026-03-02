import torch

# Global configuration for training and evaluation
MODEL_NAME = "BiLSTM_Attention"

# Training hyperparameters
SEED = 42
MAX_LEN = 25
BATCH_SIZE = 32
EMBED_DIM = 100
LR = 1e-4
EPOCHS = 13

# Data and checkpoint paths
DATA_PATH = "data/train_binary.csv"
CHECKPOINT_DIR = "artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")