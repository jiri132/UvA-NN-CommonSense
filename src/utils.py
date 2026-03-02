import torch
import random
import numpy as np
import os
import json
from datetime import datetime

def set_seed(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_experiment(model, optimizer, vocab, seed, max_len, config, artifacts_dir="artifacts"):
    """
    Save model, config, and training log in structured directories:
    artifacts/<model_name>/<entry_timestamp>/
    """
    model_name = config.get("model_name", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry_dir = os.path.join(artifacts_dir, model_name, timestamp)
    os.makedirs(entry_dir, exist_ok=True)

    model_path = os.path.join(entry_dir, "model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocab": vocab,
        "seed": seed,
        "max_len": max_len,
        "model_name": model_name
    }, model_path)

    config_path = os.path.join(entry_dir, "config.json")
    config_to_save = config.copy()
    config_to_save["torch_version"] = torch.__version__
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    log_path = os.path.join(entry_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({"epoch_metrics": []}, f, indent=2)

    print(f"Saved checkpoint to {model_path}")
    print(f"Saved config to {config_path}")
    print(f"Saved training log to {log_path}")

    return entry_dir