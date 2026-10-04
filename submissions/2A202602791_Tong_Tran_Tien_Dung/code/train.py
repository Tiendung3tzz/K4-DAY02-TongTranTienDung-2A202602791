"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Implementation for the submitted solution. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pandas as pd
import torch

import dataset
import losses
import model as model_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval import compute_metrics, save_predictions  # noqa: E402

SUBMISSION_DIR = Path(__file__).resolve().parent.parent

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = str(SUBMISSION_DIR / "runs")
    pred_dir: str = str(SUBMISSION_DIR / "predictions")
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    Notes: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). Implemented."""
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). Implemented.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    total = cfg.epochs * steps_per_epoch
    warmup = min(total, round(cfg.warmup_epochs * steps_per_epoch))
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Notes:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        import copy
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def update(self, model) -> None:
        with torch.no_grad():
            for avg, current in zip(self.model.parameters(), model.parameters()):
                avg.lerp_(current.detach(), 1 - self.decay)
            for avg, current in zip(self.model.buffers(), model.buffers()):
                avg.copy_(current)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    Notes:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    model.train()
    if cfg.init == "frozen":
        model.eval()  # frozen BatchNorm statistics must remain fixed
        model.get_classifier().train()
    total_loss, total_images = 0.0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per iteration, after optimizer.step
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.detach()) * len(images)
        total_images += len(images)
    if not total_images:
        raise ValueError("Empty training loader")
    return {"train_loss": total_loss / total_images, "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    Notes: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    names, labels_all, logits_all = [], [], []
    loss_sum, n = 0.0, 0
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels)) * len(images)
            n += len(images)
            names.extend(filenames)
            labels_all.append(labels.cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
    if not n:
        raise ValueError("Empty evaluation loader")
    return names, np.concatenate(labels_all), np.concatenate(logits_all), loss_sum / n


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Notes: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["macro_f1_val"] for row in history], label="Macro-F1 val")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _finished_history(out: Path, cfg: Config):
    """Recover a finished training loop interrupted during final checkpoint loading."""
    history_path, checkpoint_path = out / "history.csv", out / "best.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return None
    history = pd.read_csv(history_path)
    if (len(history) != cfg.epochs or
            history["epoch"].tolist() != list(range(1, cfg.epochs + 1))):
        return None
    best_row = history.iloc[history["macro_f1_val"].to_numpy().argmax()]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("epoch") != int(best_row["epoch"]) or
            not np.isclose(checkpoint.get("macro_f1_val"), best_row["macro_f1_val"])):
        return None
    return history.to_dict("records"), float(best_row["macro_f1_val"]), int(best_row["epoch"])


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Trình tự xử lý:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    if cfg.save_test_predictions and pred_path(cfg, "test").exists():
        raise FileExistsError(f"Test predictions already exist: {pred_path(cfg, 'test')}")
    set_seed(cfg.seed)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != asdict(cfg):
        raise ValueError(f"{cfg.exp_id} seed {cfg.seed}: existing configuration differs")
    finished = _finished_history(out, cfg)
    if finished is not None:
        print(f"{cfg.exp_id} seed={cfg.seed}: reusing {cfg.epochs} saved epochs after interrupted finalization")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (out / "split_check.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
                                       dataset.build_transforms(True, cfg.img_size, cfg.aug),
                                       cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_transform = dataset.build_transforms(False, cfg.img_size)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_transform,
                                     cfg.batch_size, False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_lib.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    import timm
    import torchvision
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "timm": timm.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                   "weight_tag": getattr(net, "weight_tag", "unknown")}
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    counts = train_df["Label"].value_counts().reindex(range(9), fill_value=0).to_numpy()
    weight = None
    if cfg.loss == "ce_weighted":
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    try:
        gmacs = model_lib.count_gmacs(net, cfg.img_size)
    except Exception as exc:
        print(f"GMAC measurement unavailable for {cfg.backbone}: {exc}")
        gmacs = None
    history, best_f1, best_epoch = finished if finished is not None else ([], -1.0, None)
    epoch_times = [row["epoch_seconds"] for row in history]
    train_times = [row["train_seconds"] for row in history]
    for epoch in range(len(history) + 1, cfg.epochs + 1):
        start = time.perf_counter()
        train_result = train_one_epoch(net, train_loader, criterion, optimizer,
                                       scheduler, scaler, cfg, device, ema)
        train_seconds = time.perf_counter() - start
        eval_model = ema.model if ema is not None else net
        _, y_val, z_val, val_loss = evaluate(eval_model, val_loader, criterion, device)
        probs = torch.softmax(torch.as_tensor(z_val), dim=1).numpy()
        metrics = compute_metrics(y_val, probs.argmax(1), probs)
        seconds = time.perf_counter() - start
        epoch_times.append(seconds)
        train_times.append(train_seconds)
        row = {"epoch": epoch, **train_result, "val_loss": val_loss,
               "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
               "train_seconds": train_seconds, "epoch_seconds": seconds}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch}: {row}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({"state_dict": eval_model.state_dict(), "epoch": epoch,
                        "macro_f1_val": best_f1}, out / "best.pt")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model_lib.load_checkpoint_state(net, checkpoint["state_dict"])
    names, y_val, z_val, _ = evaluate(net, val_loader, criterion, device)
    val_probs = torch.softmax(torch.as_tensor(z_val), dim=1).numpy()
    np.savez_compressed(out / "val_logits.npz", filenames=np.asarray(names), y_true=y_val, logits=z_val)
    save_predictions(pred_path(cfg, "val"), names, y_val, val_probs)
    result = {"best_epoch": best_epoch, "macro_f1_val": best_f1,
              "top1_val": compute_metrics(y_val, val_probs.argmax(1), val_probs)["top1"],
              "seconds_per_epoch": float(np.mean(epoch_times)),
              "train_seconds_per_epoch": float(np.mean(train_times)),
              "params_m": model_lib.count_params(net), "gmacs": gmacs,
              "weight_tag": getattr(net, "weight_tag", "unknown")}
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, val_transform,
                                          cfg.batch_size, False, num_workers=cfg.num_workers)
        names, y_test, z_test, _ = evaluate(net, test_loader, criterion, device)
        test_probs = torch.softmax(torch.as_tensor(z_test), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), names, y_test, test_probs)
        np.savez_compressed(out / "test_logits.npz", filenames=np.asarray(names), y_true=y_test, logits=z_test)
    plot_curves(history, SUBMISSION_DIR / "curves" / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png",
                f"{cfg.exp_id} {cfg.backbone} seed {cfg.seed}")
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    Notes: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    defaults = Config()
    hints = get_type_hints(Config)
    allowed = {field.name for field in fields(Config)}
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if key not in allowed:
            raise ValueError(f"Unknown Config key: {key}")
        annotation = hints[key]
        args = get_args(annotation)
        nullable = type(None) in args
        kind = next((item for item in args if item is not type(None)), annotation)
        if value.lower() in {"none", "null"}:
            if not nullable:
                raise ValueError(f"{key} cannot be None")
            converted = None
        elif kind is bool:
            if value.lower() not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"Invalid boolean for {key}: {value}")
            converted = value.lower() in {"true", "1", "yes"}
        else:
            converted = kind(value)
        overrides[key] = converted
    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    Notes: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
