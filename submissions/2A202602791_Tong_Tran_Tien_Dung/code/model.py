"""Backbone construction and optimizer parameter groups."""
from __future__ import annotations

import copy

import torch

SUGGESTED_BACKBONES = {"resnet50": "resnet50", "resnext50": "resnext50_32x4d",
                       "convnext_tiny": "convnext_tiny", "deit_small": "deit_small_patch16_224",
                       "swin_tiny": "swin_tiny_patch4_window7_224",
                       "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100"}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    import timm
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Unknown initialization: {init}")
    model = timm.create_model(SUGGESTED_BACKBONES.get(name, name),
                              pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    model.weight_tag = ((model.pretrained_cfg.get("hf_hub_id") or
                         model.pretrained_cfg.get("url") or
                         model.pretrained_cfg.get("architecture") or name)
                        if pretrained and init != "scratch" else "scratch")
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    classifier = model.get_classifier()
    head_ids = {id(param) for param in classifier.parameters()}
    if not head_ids:
        raise ValueError("Model classifier has no trainable parameters")
    for param in model.parameters():
        param.requires_grad_(id(param) in head_ids)
    model._backbone_frozen = True
    model.eval()
    classifier.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = {id(param) for param in model.get_classifier().parameters()}
    backbone_decay, backbone_no_decay, head = [], [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_ids:
            head.append(param)
        elif param.ndim <= 1:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)
    return [group for group in (
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay}) if group["params"]]


def count_params(model) -> float:
    return sum(param.numel() for param in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """THOP MAC estimate; install thop on Kaggle before benchmarking."""
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Install thop to measure GMAC: pip install thop") from exc
    # THOP registers total_ops/total_params buffers on the profiled modules.
    # Profile a separate CPU copy so they never enter a training checkpoint.
    profiled = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        macs, _ = profile(profiled, inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)
    del profiled
    return float(macs / 1e9)


def load_checkpoint_state(model, state_dict) -> None:
    """Load weights, ignoring only THOP counters from older checkpoints."""
    clean = {key: value for key, value in state_dict.items()
             if key.rsplit(".", 1)[-1] not in {"total_ops", "total_params"}}
    model.load_state_dict(clean)
