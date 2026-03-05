import torch
from src.metrics import evaluate_model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, threads=4):

    torch.set_num_threads(threads)

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        batch_count = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count

        print("\n===================================")
        print(f"Epoch {epoch+1}/{epochs}")
        print("===================================")
        print(f"Training Loss: {avg_loss:.4f}")

        # run validation metrics
        metrics = evaluate_model(model, val_loader, device)

        print("-----------------------------------")
        print(f"Val Accuracy : {metrics['accuracy']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall   : {metrics['recall']:.4f}")
        print(f"Val F1 Score : {metrics['f1']:.4f}")
        print("===================================\n")


def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5)
            labels_bool = (labels > 0.5)
            correct += (preds == labels_bool).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Val Accuracy: {acc:.4f}")
    return acc