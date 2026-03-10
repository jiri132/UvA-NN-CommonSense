import torch

# Global configuration for training and evaluation
MODEL_NAME = "Classifier_Attention_Bernoulli"

# Training hyperparameters
SEED = 42
MAX_LEN = 50
BATCH_SIZE = 64
EMBED_DIM = 250
HIDDEN_DIM = 256
LR = 3e-4
EPOCHS = 10

# Data and checkpoint paths
DATA_PATH = "data/train_binary.csv"
CHECKPOINT_DIR = "artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")