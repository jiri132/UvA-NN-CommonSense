import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def compute_metrics(all_labels, all_preds):

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

    return metrics


def evaluate_model(model, dataloader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)

    print("\nValidation Metrics")
    print("-------------------")

    for k, v in metrics.items():

        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

    print("\nDetailed report")
    print(classification_report(all_labels, all_preds))

    return metrics

def compute_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))