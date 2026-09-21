"""
data_loader.py
Builds train/val/test DataLoaders using torchvision.ImageFolder.

Expected directory structure:
data/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
├── val/
│   ├── class_a/
│   └── ...
└── test/
    ├── class_a/
    └── ...
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet stats — required since we're using an ImageNet-pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_transforms():
    """Returns (train_transforms, eval_transforms).
    Augmentation is applied ONLY to training data.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, eval_transforms


def get_dataloaders(data_dir="data", batch_size=32, num_workers=2):
    """
    data_dir must contain train/, val/, test/ subfolders in ImageFolder format.
    Returns: train_loader, val_loader, test_loader, class_names
    """
    train_transforms, eval_transforms = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transforms)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders("data")