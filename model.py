
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def _new_resnet18(num_classes):
    """Loads ImageNet-pretrained ResNet18 and swaps the final layer."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_frozen_model(num_classes):
    """
    Experiment 1: Feature extraction.
    Entire backbone frozen — only the new head (`fc`) is trainable.
    """
    model = _new_resnet18(num_classes)

    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name  # True only for the new head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Frozen model] Trainable params: {trainable:,} / {total:,}")

    return model


def build_partial_finetune_model(num_classes):
    """
    Experiment 2: Partial fine-tuning.
    layer1, layer2 stay frozen (generic low-level features).
    layer3, layer4, and fc are unfrozen (task-specific fine-tuning).
    """
    model = _new_resnet18(num_classes)

    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Partial fine-tune model] Trainable params: {trainable:,} / {total:,}")

    return model


def get_optimizer(model, mode="frozen", head_lr=1e-3, backbone_lr=1e-5):
    """
    mode="frozen": single param group (only head is trainable anyway)
    mode="finetune": discriminative learning rates — backbone gets a much
                      smaller LR than the head, since it's already well-trained
                      and we only want small adjustments.
    """
    if mode == "frozen":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr
        )

    elif mode == "finetune":
        backbone_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("fc")]
        head_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("fc")]

        return torch.optim.Adam([
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ])

    else:
        raise ValueError("mode must be 'frozen' or 'finetune'")