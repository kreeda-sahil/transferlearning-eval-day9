import time
import copy
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, optimizer, num_epochs=15,
                 patience=4, device=None, label="model"):
    """
    Returns: trained model (best val weights restored), history dict
    history = {"train_loss": [...], "val_loss": [...],
               "train_acc": [...], "val_acc": [...]}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        start = time.time()

        # Train phase 
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validation phase 
        model.eval()
        val_running_loss, val_running_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_running_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"[{label}] Epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)")

        # ---- Early stopping check ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{label}] Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_weights)
    return model, history