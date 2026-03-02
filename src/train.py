import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")
        validate_model(model, val_loader, device)

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