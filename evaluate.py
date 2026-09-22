import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)


@torch.no_grad()
def get_predictions(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")
    return cm


def evaluate_model(model, test_loader, class_names, label="model", save_dir="results"):
    """
    Full evaluation: prints classification report, saves confusion matrix plot,
    and returns a dict of summary metrics for cross-experiment comparison.
    """
    y_true, y_pred = get_predictions(model, test_loader)

    print(f"\n=== Classification Report: {label} ===")
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print(report_str)

    cm = plot_confusion_matrix(
        y_true, y_pred, class_names,
        title=f"Confusion Matrix — {label}",
        save_path=f"{save_dir}/{label}_confusion_matrix.png"
    )

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    summary = {
        "model": label,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }
    return summary, cm


def save_comparison_csv(summaries, save_path="results/metrics_comparison.csv"):
    df = pd.DataFrame(summaries)
    df.to_csv(save_path, index=False)
    print(f"\nSaved comparison table -> {save_path}")
    print(df.to_string(index=False))
    return df


def plot_training_curves(history_frozen, history_finetune, save_path="results/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_frozen["train_loss"], label="Frozen - train")
    axes[0].plot(history_frozen["val_loss"], label="Frozen - val")
    axes[0].plot(history_finetune["train_loss"], label="Fine-tuned - train", linestyle="--")
    axes[0].plot(history_finetune["val_loss"], label="Fine-tuned - val", linestyle="--")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_frozen["train_acc"], label="Frozen - train")
    axes[1].plot(history_frozen["val_acc"], label="Frozen - val")
    axes[1].plot(history_finetune["train_acc"], label="Fine-tuned - train", linestyle="--")
    axes[1].plot(history_finetune["val_acc"], label="Fine-tuned - val", linestyle="--")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves -> {save_path}")