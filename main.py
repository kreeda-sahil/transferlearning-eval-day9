"""
main.py
Runs the full experiment: frozen feature-extraction vs partial fine-tuning,
evaluates both on the same test set, and saves all comparison artifacts.

Usage:
    python src/main.py
"""

import os
from data_loader import get_dataloaders
from model import build_frozen_model, build_partial_finetune_model, get_optimizer
from train import train_model
from evaluate import evaluate_model, save_comparison_csv, plot_training_curves

os.makedirs("results", exist_ok=True)


def main():
    # 1. Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="data", batch_size=32
    )
    num_classes = len(class_names)

    # 2. Experiment 1 — Frozen backbone (feature extraction)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Frozen backbone (feature extraction)")
    print("=" * 60)
    frozen_model = build_frozen_model(num_classes)
    frozen_optimizer = get_optimizer(frozen_model, mode="frozen", head_lr=1e-3)
    frozen_model, history_frozen = train_model(
        frozen_model, train_loader, val_loader, frozen_optimizer,
        num_epochs=15, patience=4, label="frozen"
    )
    summary_frozen, _ = evaluate_model(
        frozen_model, test_loader, class_names, label="frozen"
    )

    # 3. Experiment 2 — Partial fine-tuning (layer3, layer4, fc unfrozen)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Partial fine-tuning")
    print("=" * 60)
    finetune_model = build_partial_finetune_model(num_classes)
    finetune_optimizer = get_optimizer(
        finetune_model, mode="finetune", head_lr=1e-3, backbone_lr=1e-5
    )
    finetune_model, history_finetune = train_model(
        finetune_model, train_loader, val_loader, finetune_optimizer,
        num_epochs=15, patience=4, label="finetuned"
    )
    summary_finetune, _ = evaluate_model(
        finetune_model, test_loader, class_names, label="finetuned"
    )

    # 4. Compare
    save_comparison_csv([summary_frozen, summary_finetune])
    plot_training_curves(history_frozen, history_finetune)

    print("\nDone. Check the results/ folder for:")
    print("  - frozen_confusion_matrix.png")
    print("  - finetuned_confusion_matrix.png")
    print("  - training_curves.png")
    print("  - metrics_comparison.csv")


if __name__ == "__main__":
    main()