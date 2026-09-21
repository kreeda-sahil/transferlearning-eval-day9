# Transfer Learning Evaluation - Day 9

This project evaluates transfer learning for multi-class scene image classification using a pretrained ResNet18 model in PyTorch. It compares two common transfer-learning strategies on the same dataset:

1. Frozen backbone feature extraction
2. Partial fine-tuning of deeper ResNet18 layers

The goal is to check whether unfreezing later layers of a pretrained CNN improves classification performance compared with training only a new final classification head.

## Dataset

The project expects the dataset in `torchvision.datasets.ImageFolder` format:

```text
data/
  train/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
  val/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
  test/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
```

From the latest run:

| Split | Images |
| --- | ---: |
| Train | 12,632 |
| Validation | 1,402 |
| Test | 3,000 |

The six classes are `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

## Approach

The pipeline is split across small modules:

- `data_loader.py` builds train, validation, and test DataLoaders.
- `model.py` creates the pretrained ResNet18 models and controls which layers are trainable.
- `train.py` contains the training loop, validation tracking, and early stopping.
- `evaluate.py` creates classification reports, confusion matrices, metric summaries, and training curves.
- `main.py` runs both experiments end to end.

Training images use augmentation such as random resized crop, horizontal flip, color jitter, and rotation. Validation and test images use deterministic resize and center crop. All images are normalized with ImageNet mean and standard deviation because the backbone is pretrained on ImageNet.

## Experiments

### Experiment 1: Frozen Backbone

The ResNet18 backbone is fully frozen and only the final `fc` classification layer is trained.

- Trainable parameters: 3,078 / 11,179,590
- Early stopped at epoch 13
- Test accuracy: 90.13%
- Macro F1: 90.30%

### Experiment 2: Partial Fine-Tuning

The earlier ResNet18 layers remain frozen, while `layer3`, `layer4`, and `fc` are trainable. The optimizer uses a smaller learning rate for the pretrained backbone and a larger learning rate for the new classification head.

- Trainable parameters: 10,496,518 / 11,179,590
- Early stopped at epoch 9
- Test accuracy: 93.07%
- Macro F1: 93.17%

## Results Summary

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen backbone | 0.9013 | 0.9027 | 0.9037 | 0.9030 | 0.9011 |
| Partial fine-tuning | 0.9307 | 0.9315 | 0.9324 | 0.9317 | 0.9306 |

Partial fine-tuning performed better overall, improving test accuracy by about 2.93 percentage points over the frozen-backbone setup. The result suggests that the pretrained ImageNet features are already strong for this scene-classification task, but adapting the deeper layers gives the model a useful performance boost.

