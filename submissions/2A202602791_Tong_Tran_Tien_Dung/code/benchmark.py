"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Implementation for the submitted solution.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    Notes:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("Use at least 10 warmup and 50 measured iterations")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(np.mean(times)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    Notes:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp or fp16")
    device = torch.device(device)
    if dtype == "fp16" and device.type != "cuda":
        raise ValueError("fp16 benchmark requires CUDA")
    model = model.to(device).eval()
    if dtype == "fp16":
        model = model.half()
    else:
        model = model.float()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device,
                         dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                   enabled=dtype == "amp" and device.type == "cuda"):
            model(images)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    result = bench(forward, warmup, iters, sync)
    result.update({"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "dtype": dtype, "batch": batch_size, "img_size": img_size,
                   "images_per_s": batch_size / (result["p50"] / 1000),
                   "torch": torch.__version__, "preprocessing_included": False})
    return result


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). Notes: đo thật, so với K * p50."""
    if k_views < 1:
        raise ValueError("k_views must be positive")
    batch_size = kw.pop("batch_size", 1)
    img_size = kw.pop("img_size", 224)
    dtype = kw.pop("dtype", "fp32")
    device = torch.device(kw.pop("device", "cuda"))
    warmup = kw.pop("warmup", 10)
    iters = kw.pop("iters", 100)
    if kw:
        raise ValueError(f"Unknown options: {list(kw)}")
    single = latency_report(model, batch_size, img_size, dtype, str(device), warmup, iters)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                   enabled=dtype == "amp" and device.type == "cuda"):
            for _ in range(k_views):
                model(x)
    measured = bench(forward, warmup, iters,
                     torch.cuda.synchronize if device.type == "cuda" else None)
    measured.update({"k_views": k_views, "predicted_p50": single["p50"] * k_views,
                     "gpu": single["gpu"], "dtype": dtype, "batch": batch_size,
                     "img_size": img_size, "torch": torch.__version__})
    return measured


def method_latency(model, method: str, batch_size: int = 1, img_size: int = 224,
                   temperature: float | None = None, device: str = "cuda",
                   warmup: int = 10, iters: int = 50) -> dict:
    """Measure the actual view/aggregation path, excluding image file loading."""
    from inference import forward_probs
    device = torch.device(device)
    model = model.to(device).float().eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    def forward():
        with torch.inference_mode():
            forward_probs(model, images, method, temperature)
    result = bench(forward, warmup, iters,
                   torch.cuda.synchronize if device.type == "cuda" else None)
    result.update({"method": method, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "dtype": "fp32", "batch": batch_size, "img_size": img_size,
                   "images_per_s": batch_size / (result["p50"] / 1000),
                   "torch": torch.__version__, "preprocessing_included": False,
                   "gpu_view_ops_included": True})
    return result
