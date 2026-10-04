"""Classification losses and batch level Mixup/CutMix."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted requires a train-only weight tensor")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Unknown loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """PyTorch CE uses eps/K probability for every class."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, target):
        return self.loss(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -(1 - pt).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.numel() != 9 or torch.any(counts <= 0):
        raise ValueError("counts must contain nine positive TRAIN class counts")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    weights = 1 / counts if beta == 0 else (1 - beta) / (1 - beta ** counts)
    return weights / weights.mean()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0 or mode not in {"mixup", "cutmix"}:
        raise ValueError("alpha must be positive and mode must be mixup or cutmix")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        _, _, height, width = x.shape
        cut_w = int(width * np.sqrt(1 - lam))
        cut_h = int(height * np.sqrt(1 - lam))
        cx, cy = np.random.randint(width), np.random.randint(height)
        x1, x2 = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
        y1, y2 = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (width * height)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
