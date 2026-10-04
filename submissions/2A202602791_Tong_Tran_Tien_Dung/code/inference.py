"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Implementation for the submitted solution.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Notes: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    device = torch.device(device)
    model.eval()
    names, labels_all, logits_all = [], [], []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            transformed = view(images) if view is not None else images
            if isinstance(transformed, (list, tuple)):
                logits = torch.stack([model(batch) for batch in transformed]).mean(0)
            else:
                logits = model(transformed)
            names.extend(filenames)
            labels_all.append(labels.numpy())
            logits_all.append(logits.float().cpu().numpy())
    return names, np.concatenate(labels_all), np.concatenate(logits_all)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). Notes: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. Implemented."""
    height, width = x.shape[-2:]
    if crop > min(height, width) or crop < 1:
        raise ValueError("crop must fit inside input")
    locations = [(0, 0), (0, width - crop), (height - crop, 0),
                 (height - crop, width - crop), ((height - crop) // 2, (width - crop) // 2)]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in locations]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. Implemented.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    return [F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False)
            for size in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    Notes: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if space not in {"prob", "logit"} or not logits_per_view:
        raise ValueError("Need views and space='prob' or 'logit'")
    z = np.stack([np.asarray(view, dtype=np.float64) for view in logits_per_view])
    if space == "logit":
        z = z.mean(axis=0)
        z -= z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)
    z -= z.max(axis=2, keepdims=True)
    exp = np.exp(z)
    probs = (exp / exp.sum(axis=2, keepdims=True)).mean(axis=0)
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). Implemented.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    if not list_of_probs:
        raise ValueError("No model probabilities")
    probs = np.stack([np.asarray(item, dtype=np.float64) for item in list_of_probs]).mean(axis=0)
    if np.any(probs < 0) or np.any(probs.sum(axis=1) <= 0):
        raise ValueError("Invalid probabilities")
    return probs / probs.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Notes: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). Implemented."""
    if T <= 0:
        raise ValueError("Temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def forward_probs(model, x, method: str = "I00", temperature: float | None = None,
                  crop: int = 192):
    """Differentiable forward for the val-selected inference methods.

    I00: one view; I01: horizontal flip, mean probabilities;
    I02: five crops resized back to input size, mean probabilities;
    I03: horizontal flip, mean logits. Calibration is applied after aggregation.
    """
    if method not in {"I00", "I01", "I02", "I03"}:
        raise ValueError(f"Unknown inference method: {method}")
    if method == "I02":
        crops = views_multicrop(x, crop)
        logits = [model(F.interpolate(part, size=x.shape[-2:], mode="bilinear",
                                      align_corners=False)) for part in crops]
        probs = torch.stack([part.softmax(dim=1) for part in logits]).mean(dim=0)
    else:
        base = model(x)
        if method == "I00":
            probs = base.softmax(dim=1)
        else:
            flipped = model(view_hflip(x))
            probs = ((base + flipped) / 2).softmax(dim=1) if method == "I03" else (
                base.softmax(dim=1) + flipped.softmax(dim=1)) / 2
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        probs = (probs.clamp_min(1e-12).log() / temperature).softmax(dim=1)
    return probs


def predict_probs(model, loader, device, method: str = "I00",
                  temperature: float | None = None, crop: int = 192):
    """Return filenames, labels, probabilities in the loader's fixed order."""
    device = torch.device(device)
    model.eval()
    names, labels_all, probs_all = [], [], []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            probs = forward_probs(model, images.to(device), method, temperature, crop)
            names.extend(filenames)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(probs.float().cpu().numpy())
    if not names:
        raise ValueError("Empty prediction loader")
    return names, np.concatenate(labels_all), np.concatenate(probs_all)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Notes:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    fused = copy.deepcopy(model).eval()
    def visit(module):
        children = list(module.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
        for child in module.children():
            visit(child)
    visit(fused)
    return fused
