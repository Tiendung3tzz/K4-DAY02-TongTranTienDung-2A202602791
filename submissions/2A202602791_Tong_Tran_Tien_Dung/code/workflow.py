"""Kaggle orchestration for val-only ablations and inference comparisons."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark
import dataset
import inference
import model as model_lib
import train
from eval import check_against_csv, compute_metrics, load_group, read_pred, save_predictions

ROOT = train.SUBMISSION_DIR
RESULTS = ROOT / "results.xlsx"


def write_sheet(frame: pd.DataFrame, name: str) -> None:
    """Replace one sheet while preserving any earlier stages in the workbook."""
    if RESULTS.exists():
        with pd.ExcelWriter(RESULTS, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            frame.to_excel(writer, sheet_name=name, index=False)
    else:
        with pd.ExcelWriter(RESULTS, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name=name, index=False)


def run_cached(cfg: train.Config) -> dict:
    """Reuse only a fully finished run with exactly the same configuration."""
    folder = train.run_dir(cfg)
    config_path = folder / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != asdict(cfg):
            raise ValueError(f"{cfg.exp_id} seed {cfg.seed}: existing configuration differs")
    summary_path = folder / "summary.json"
    if summary_path.exists() and (folder / "best.pt").exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if cfg.save_test_predictions:
        raise ValueError("Screening runs must never evaluate test")
    return train.run(cfg)


def _val_class_metrics(cfg: train.Config) -> dict:
    saved = np.load(train.run_dir(cfg) / "val_logits.npz")
    y, z = saved["y_true"], saved["logits"]
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return compute_metrics(y, probs.argmax(1), probs)


def run_ablation(backbone: str, images_dir: str | Path, labels_dir: str | Path,
                 seed: int = 0) -> pd.DataFrame:
    """T00 plus one-factor A/B/C trials and one B+C combination."""
    base = train.Config(exp_id="T00", backbone=backbone, seed=seed,
                        images_dir=str(images_dir), labels_dir=str(labels_dir),
                        save_test_predictions=False)
    variants = [
        ("T00", "baseline", {}, "T00 recipe"),
        ("T01", "A_init", {"init": "frozen"}, "frozen backbone"),
        ("T02", "A_init", {"init": "scratch"}, "train from scratch"),
        ("T03", "B_aug", {"aug": "color"}, "color jitter"),
        ("T04", "B_aug", {"mix": "cutmix"}, "CutMix alpha=1"),
        ("T05", "C_loss", {"loss": "ls", "label_smoothing": 0.1}, "label smoothing 0.1"),
        ("T06", "C_loss", {"loss": "focal"}, "focal gamma=2"),
    ]
    rows = []
    baseline_f1 = None
    for exp_id, axis, changes, note in variants:
        cfg = replace(base, exp_id=exp_id, **changes)
        if exp_id != "T00":
            changed = {key for key in asdict(base) if getattr(base, key) != getattr(cfg, key)}
            if changed != {"exp_id", *changes}:
                raise AssertionError(f"{exp_id} changes multiple factors: {changed}")
        summary = run_cached(cfg)
        if exp_id == "T00":
            baseline_f1 = summary["macro_f1_val"]
        metrics = _val_class_metrics(cfg)
        rows.append({"exp_id": exp_id, "backbone": backbone, "axis": axis,
                     "difference_from_T00": json.dumps(changes), "seed": seed,
                     "macro_f1_val": summary["macro_f1_val"],
                     "top1_val": summary["top1_val"],
                     "delta_vs_T00": summary["macro_f1_val"] - baseline_f1,
                     "f1_chinee_apple_val": float(metrics["f1"][0]),
                     "f1_snake_weed_val": float(metrics["f1"][7]),
                     "note": note + "; one screening seed"})
        write_sheet(pd.DataFrame(rows), "Training")
    # Best nonbaseline variant in each axis, selected only from val.
    aug_row = max((row for row in rows if row["axis"] == "B_aug"), key=lambda row: row["macro_f1_val"])
    loss_row = max((row for row in rows if row["axis"] == "C_loss"), key=lambda row: row["macro_f1_val"])
    aug_change = next(changes for eid, _, changes, _ in variants if eid == aug_row["exp_id"])
    loss_change = next(changes for eid, _, changes, _ in variants if eid == loss_row["exp_id"])
    combo = {**aug_change, **loss_change}
    cfg = replace(base, exp_id="T07", **combo)
    summary = run_cached(cfg)
    metrics = _val_class_metrics(cfg)
    rows.append({"exp_id": "T07", "backbone": backbone, "axis": "B_plus_C",
                 "difference_from_T00": json.dumps(combo), "seed": seed,
                 "macro_f1_val": summary["macro_f1_val"], "top1_val": summary["top1_val"],
                 "delta_vs_T00": summary["macro_f1_val"] - baseline_f1,
                 "f1_chinee_apple_val": float(metrics["f1"][0]),
                 "f1_snake_weed_val": float(metrics["f1"][7]),
                 "note": f"Combined best tested B ({aug_row['exp_id']}) and C ({loss_row['exp_id']}); one seed"})
    frame = pd.DataFrame(rows)
    write_sheet(frame, "Training")
    return frame


def best_training_config(frame: pd.DataFrame, seed: int = 0) -> train.Config:
    """Return the highest-val-F1 screening configuration, earliest ID on ties."""
    chosen = frame.sort_values(["macro_f1_val", "exp_id"], ascending=[False, True]).iloc[0]
    cfg_path = ROOT / "runs" / str(chosen["exp_id"]) / f"seed{seed}" / "config.json"
    return train.Config(**json.loads(cfg_path.read_text(encoding="utf-8")))


def load_best_model(cfg: train.Config, device: str | torch.device):
    net = model_lib.build_model(cfg.backbone, pretrained=False, init="scratch",
                                drop_rate=cfg.drop_rate)
    checkpoint = torch.load(train.run_dir(cfg) / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["state_dict"])
    return net.to(device).eval()


def _loader(cfg: train.Config, split: str):
    frames = dataset.load_split(cfg.labels_dir, cfg.fold)
    frame = {"val": frames[1], "test": frames[2]}[split]
    return dataset.make_loader(frame, cfg.images_dir,
                               dataset.build_transforms(False, cfg.img_size),
                               cfg.batch_size, False, num_workers=cfg.num_workers)


def compare_inference(cfg: train.Config, large_batch: int = 16):
    """I00/I01/I02/I03/I07 on val; all timing paths measured on GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU for inference timing")
    net = load_best_model(cfg, "cuda")
    loader = _loader(cfg, "val")
    rows, latency_rows, predictions = [], [], {}
    names_ref = labels_ref = None
    for exp_id, method, description in [
        ("I00", "I00", "single center crop"),
        ("I01", "I01", "horizontal flip, probability mean"),
        ("I02", "I02", "five resized crops, probability mean"),
        ("I03", "I03", "horizontal flip, logit mean"),
    ]:
        names, labels, probs = inference.predict_probs(net, loader, "cuda", method)
        if names_ref is None:
            names_ref, labels_ref = names, labels
        elif names != names_ref or not np.array_equal(labels, labels_ref):
            raise ValueError("Validation order changed between inference methods")
        predictions[exp_id] = probs
        metrics = compute_metrics(labels, probs.argmax(1), probs)
        timed = {}
        for batch in (1, large_batch):
            measurement = benchmark.method_latency(net, method, batch_size=batch,
                                                   img_size=cfg.img_size, iters=50)
            latency_rows.append({"config": exp_id, "gpu": measurement["gpu"],
                                 "dtype": measurement["dtype"], "batch": batch,
                                 "fused_bn": False, "p50_ms": measurement["p50"],
                                 "p95_ms": measurement["p95"], "p99_ms": measurement["p99"],
                                 "images_per_s": measurement["images_per_s"],
                                 "torch": measurement["torch"], "preprocessing_included": False})
            timed[batch] = measurement
        rows.append({"exp_id": exp_id, "method": description,
                     "checkpoint": str(train.run_dir(cfg) / "best.pt"),
                     "k": 5 if method == "I02" else 2 if method in {"I01", "I03"} else 1,
                     "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
                     "ece_val": metrics["ece"], "p50_ms": timed[1]["p50"],
                     "p95_ms": timed[1]["p95"], "p99_ms": timed[1]["p99"],
                     "images_per_s": timed[1]["images_per_s"]})
        write_sheet(pd.DataFrame(rows), "Inference")
        write_sheet(pd.DataFrame(latency_rows), "Latency")
    # I07 calibrates the one-view reference, using validation data only.
    temperature = inference.fit_temperature(np.log(np.clip(predictions["I00"], 1e-12, 1)), labels_ref)
    calibrated = inference.apply_temperature(np.log(np.clip(predictions["I00"], 1e-12, 1)), temperature)
    metrics = compute_metrics(labels_ref, calibrated.argmax(1), calibrated)
    timed = {}
    for batch in (1, large_batch):
        measurement = benchmark.method_latency(net, "I00", batch_size=batch,
                                               img_size=cfg.img_size, temperature=temperature,
                                               iters=50)
        latency_rows.append({"config": "I07", "gpu": measurement["gpu"],
                             "dtype": measurement["dtype"], "batch": batch,
                             "fused_bn": False, "p50_ms": measurement["p50"],
                             "p95_ms": measurement["p95"], "p99_ms": measurement["p99"],
                             "images_per_s": measurement["images_per_s"],
                             "torch": measurement["torch"], "preprocessing_included": False})
        timed[batch] = measurement
    rows.append({"exp_id": "I07", "method": f"temperature scaling, T={temperature:.4f}",
                 "checkpoint": str(train.run_dir(cfg) / "best.pt"), "k": 1,
                 "macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
                 "ece_val": metrics["ece"], "p50_ms": timed[1]["p50"],
                 "p95_ms": timed[1]["p95"], "p99_ms": timed[1]["p99"],
                 "images_per_s": timed[1]["images_per_s"]})
    frame = pd.DataFrame(rows)
    frame["relative_cost_vs_I00"] = frame["p50_ms"] / frame.loc[frame.exp_id == "I00", "p50_ms"].iloc[0]
    write_sheet(frame, "Inference")
    write_sheet(pd.DataFrame(latency_rows), "Latency")
    best = frame.loc[frame.exp_id.isin(["I00", "I01", "I02", "I03"])].sort_values(
        ["macro_f1_val", "p50_ms"], ascending=[False, True]).iloc[0]
    best_method = str(best["exp_id"])
    best_probs = predictions[best_method]
    best_t = inference.fit_temperature(np.log(np.clip(best_probs, 1e-12, 1)), labels_ref)
    best_cal = inference.apply_temperature(np.log(np.clip(best_probs, 1e-12, 1)), best_t)
    ece_before = compute_metrics(labels_ref, best_probs.argmax(1), best_probs)["ece"]
    ece_after = compute_metrics(labels_ref, best_cal.argmax(1), best_cal)["ece"]
    recommendation = {"source_exp_id": cfg.exp_id, "method": best_method,
                      "calibrate": bool(ece_after < ece_before),
                      "val_ece_before": float(ece_before), "val_ece_after": float(ece_after),
                      "seed": cfg.seed}
    (ROOT / "val_selection_draft.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame["p50_ms"], frame["macro_f1_val"])
    for _, item in frame.iterrows():
        ax.annotate(item["exp_id"], (item["p50_ms"], item["macro_f1_val"]))
    ax.set(xlabel="Batch-1 p50 (ms)", ylabel="Macro-F1 val", title="Inference quality vs latency")
    fig.tight_layout()
    path = ROOT / "curves" / "inference_tradeoff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return frame, recommendation


def train_final_pair(selected: train.Config, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Train final recipe and T00 reference on train; select only via val."""
    if len(set(seeds)) < 3:
        raise ValueError("Final comparison requires at least three distinct seeds")
    rows = []
    for seed in seeds:
        for exp_id in ("T00", "F01"):
            cfg = (replace(selected, exp_id="F01", seed=seed, save_test_predictions=False)
                   if exp_id == "F01" else
                   train.Config(exp_id="T00", backbone=selected.backbone, seed=seed,
                                images_dir=selected.images_dir, labels_dir=selected.labels_dir,
                                save_test_predictions=False))
            summary = run_cached(cfg)
            rows.append({"exp_id": exp_id, "seed": seed, "backbone": cfg.backbone,
                         "macro_f1_val": summary["macro_f1_val"],
                         "top1_val": summary["top1_val"],
                         "best_epoch": summary["best_epoch"]})
    return pd.DataFrame(rows)


def predict_test_once(cfg: train.Config, method: str, calibrate: bool = False) -> Path:
    """Evaluate the test fold once, with an on-disk guard against a second read.

    Any temperature is fitted independently for this seed using val. If the
    process dies after test_started.json is created, manual audit is required.
    """
    if method not in {"I00", "I01", "I02", "I03"}:
        raise ValueError("Select a method validated in Step 3")
    if cfg.save_test_predictions:
        raise ValueError("Use a val-only training Config; this function controls test access")
    target = train.pred_path(cfg, "test")
    reference = str(Path(cfg.labels_dir) / f"test_subset{cfg.fold}.csv")
    folder = train.run_dir(cfg)
    lock = folder / "test_started.json"
    if target.exists():
        if not lock.exists():
            raise RuntimeError(f"Existing test file has no evaluation lock: {target}")
        recorded = json.loads(lock.read_text(encoding="utf-8"))
        if recorded["method"] != method or recorded["calibrate"] != calibrate:
            raise RuntimeError(f"Existing test file used a different method: {target}")
        check_against_csv(read_pred(str(target)), reference)
        return target
    if lock.exists():
        raise RuntimeError(f"Test was already started for {cfg.exp_id} seed {cfg.seed}; inspect {lock}")
    if not (folder / "best.pt").exists():
        raise FileNotFoundError(f"Missing val-selected checkpoint: {folder / 'best.pt'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_best_model(cfg, device)
    names_val, labels_val, raw_val = inference.predict_probs(net, _loader(cfg, "val"), device, method)
    temperature = (inference.fit_temperature(np.log(np.clip(raw_val, 1e-12, 1)), labels_val)
                   if calibrate else None)
    final_val = (inference.apply_temperature(np.log(np.clip(raw_val, 1e-12, 1)), temperature)
                 if calibrate else raw_val)
    if calibrate:
        save_predictions(train.pred_path(replace(cfg, exp_id=f"{cfg.exp_id}uncal"), "val"),
                         names_val, labels_val, raw_val)
    save_predictions(train.pred_path(cfg, "val"), names_val, labels_val, final_val)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"exp_id": cfg.exp_id, "seed": cfg.seed, "method": method,
                   "calibrate": calibrate, "temperature_fit_on_val": temperature,
                   "checkpoint": str(folder / "best.pt")}, stream, indent=2)
    # The following is the only test image pass for this exp_id and seed.
    names_test, labels_test, raw_test = inference.predict_probs(
        net, _loader(cfg, "test"), device, method)
    if calibrate:
        save_predictions(train.pred_path(replace(cfg, exp_id=f"{cfg.exp_id}uncal"), "test"),
                         names_test, labels_test, raw_test)
        probs_test = inference.apply_temperature(np.log(np.clip(raw_test, 1e-12, 1)), temperature)
    else:
        probs_test = raw_test
    save_predictions(target, names_test, labels_test, probs_test)
    check_against_csv(read_pred(str(target)), reference)
    return target


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, record in subset.iterrows():
        values = []
        for value in record:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def make_final_artifacts(selected: train.Config, method: str, calibrate: bool,
                         seeds=(0, 1, 2)) -> dict:
    """Build Final/PerClass/Summary sheets and report from actual saved predictions."""
    seeds = tuple(seeds)
    if len(set(seeds)) < 3:
        raise ValueError("At least three seeds are required")
    reference = str(Path(selected.labels_dir) / f"test_subset{selected.fold}.csv")
    val_reference = str(Path(selected.labels_dir) / f"val_subset{selected.fold}.csv")
    groups = {}
    for exp_id in ("T00", "F01"):
        paths = [str(ROOT / "predictions" / f"{exp_id}_seed{seed}_test.csv") for seed in seeds]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing test predictions: {missing}")
        groups[exp_id] = load_group(paths, reference)
    rows = []
    for exp_id, group in groups.items():
        for pred, metrics in zip(group.preds, group.metrics):
            val_path = ROOT / "predictions" / f"{exp_id}_seed{pred.seed}_val.csv"
            val_pred = read_pred(str(val_path))
            check_against_csv(val_pred, val_reference, "val")
            val_metrics = compute_metrics(val_pred.y_true, val_pred.y_pred, val_pred.probs)
            rows.append({"exp_id": exp_id, "configuration": "T00 + I00" if exp_id == "T00" else
                         f"{selected.backbone} + {selected.exp_id} recipe + {method}" +
                         (" + calibrated" if calibrate else ""),
                         "seed": pred.seed, "macro_f1_val": val_metrics["macro_f1"],
                         "macro_f1_test": metrics["macro_f1"], "top1_test": metrics["top1"],
                         "ece_test": metrics["ece"], "mean_std": ""})
        val_values = [row["macro_f1_val"] for row in rows if row["exp_id"] == exp_id
                      and isinstance(row["seed"], int)]
        rows.append({"exp_id": exp_id, "configuration": "summary",
                     "seed": "mean ± std", "macro_f1_val": float(np.mean(val_values)),
                     "macro_f1_test": group.summary["macro_f1"][0],
                     "top1_test": group.summary["top1"][0],
                     "ece_test": group.summary["ece"][0],
                     "mean_std": _format_mean_std(*group.summary["macro_f1"])})
    final_frame = pd.DataFrame(rows)
    write_sheet(final_frame, "Final")
    class_rows = []
    for exp_id, group in groups.items():
        for i, name in enumerate(dataset.CLASS_NAMES):
            class_rows.append({"exp_id": exp_id, "class": name,
                               "support_test": int(group.metrics[0]["support"][i]),
                               "precision_mean": float(group.summary["precision"][0][i]),
                               "precision_std": float(group.summary["precision"][1][i]),
                               "recall_mean": float(group.summary["recall"][0][i]),
                               "recall_std": float(group.summary["recall"][1][i]),
                               "f1_mean": float(group.summary["f1"][0][i]),
                               "f1_std": float(group.summary["f1"][1][i])})
    per_class = pd.DataFrame(class_rows)
    write_sheet(per_class, "PerClass")
    summary_rows = []
    for sheet in ("Backbones", "Training", "Inference"):
        if sheet not in pd.ExcelFile(RESULTS).sheet_names:
            raise ValueError(f"Missing prerequisite results sheet: {sheet}")
        frame = pd.read_excel(RESULTS, sheet_name=sheet)
        for _, item in frame.iterrows():
            summary_rows.append({"stage": sheet, "exp_id": item["exp_id"],
                                 "macro_f1_val": item["macro_f1_val"],
                                 "latency_batch1_p50_ms": item.get("latency_batch1_p50_ms", item.get("p50_ms", np.nan)),
                                 "backbone_or_method": item.get("backbone", item.get("method", ""))})
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_val", ascending=False).head(10)
    write_sheet(summary, "Summary")
    from openpyxl import load_workbook
    workbook = load_workbook(RESULTS)
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            letter = column[0].column_letter
            width = min(48, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
            sheet.column_dimensions[letter].width = width
    workbook.save(RESULTS)
    import matplotlib.pyplot as plt
    confusion = sum((m["confusion"] for m in groups["F01"].metrics))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(9), yticks=range(9), xticklabels=dataset.CLASS_NAMES,
           yticklabels=dataset.CLASS_NAMES, xlabel="Predicted", ylabel="True",
           title="F01 test confusion matrix, sum across seeds")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    (ROOT / "curves").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "curves" / "F01_confusion.png", dpi=160)
    plt.close(fig)
    first = groups["F01"].preds[0]
    wrong = np.flatnonzero(first.y_pred != first.y_true)
    pd.DataFrame({"Filename": first.filenames[wrong], "y_true": first.y_true[wrong],
                  "y_pred": first.y_pred[wrong],
                  "confidence": first.probs[wrong].max(axis=1)}).to_csv(
                      ROOT / "misclassified_seed0.csv", index=False)
    final = groups["F01"].summary
    base = groups["T00"].summary
    delta = final["macro_f1"][0] - base["macro_f1"][0]
    noise = max(final["macro_f1"][1], base["macro_f1"][1])
    comparison = ("Lợi thế lớn hơn std quan sát qua seed." if delta > noise else
                  "Chênh lệch không vượt std quan sát; chưa phân biệt được hai cấu hình.")
    backbone_table = _markdown_table(pd.read_excel(RESULTS, sheet_name="Backbones"),
                                     ["exp_id", "backbone", "macro_f1_val", "top1_val",
                                      "latency_batch1_p50_ms"])
    training_table = _markdown_table(pd.read_excel(RESULTS, sheet_name="Training"),
                                     ["exp_id", "axis", "macro_f1_val", "delta_vs_T00",
                                      "f1_chinee_apple_val", "f1_snake_weed_val"])
    inference_table = _markdown_table(pd.read_excel(RESULTS, sheet_name="Inference"),
                                      ["exp_id", "macro_f1_val", "ece_val", "p50_ms", "p95_ms"])
    final_table = _markdown_table(final_frame, ["exp_id", "seed", "macro_f1_val",
                                                  "macro_f1_test", "top1_test", "ece_test"])
    confusion_offdiag = confusion.copy()
    np.fill_diagonal(confusion_offdiag, 0)
    hard_pair = np.unravel_index(confusion_offdiag.argmax(), confusion_offdiag.shape)
    hard_pair_text = (f"{dataset.CLASS_NAMES[hard_pair[0]]} → "
                      f"{dataset.CLASS_NAMES[hard_pair[1]]}: {confusion_offdiag[hard_pair]} lần")
    env_path = train.run_dir(replace(selected, exp_id="F01", seed=seeds[0])) / "environment.json"
    environment = (json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {})
    env_text = ", ".join(f"{key}: {value}" for key, value in environment.items())
    report = f"""# DeepWeeds Lab Day 2 — Báo cáo thực nghiệm

## 1. Tóm tắt

Bài toán phân loại DeepWeeds 9 lớp, dùng fold 0 do tác giả cung cấp. Cấu hình chung kết {selected.backbone}, công thức {selected.exp_id}, suy luận {method}{' + temperature scaling' if calibrate else ''}. Macro-F1 test qua {len(seeds)} seed: {_format_mean_std(*final['macro_f1'])}; mốc T00 + I00: {_format_mean_std(*base['macro_f1'])}; chênh lệch {delta:+.4f}. Top-1 test chung kết: {_format_mean_std(*final['top1'])}.

## 2. Dữ liệu và thiết lập

Fold 0, train/val/test tách sẵn và không gộp. Xem số đếm, biểu đồ lớp, ảnh mẫu và kiểm tra pipeline trong notebook Bước 0. Cấu hình và log từng lần chạy nằm trong `runs/<exp_id>/seed<k>/`. Môi trường lần chạy: {env_text or 'xem runs/F01/seed0/environment.json'}. **Bổ sung tại đây:** link notebook và nhận xét EDA từ dữ liệu thật.

## 3. So sánh backbone

Nguồn: sheet `Backbones`, biểu đồ `curves/backbone_tradeoff.png`.

{backbone_table}

**Bổ sung phân tích:** mô hình nào hội tụ nhanh, mô hình nào cân bằng F1 và độ trễ, vì sao chọn backbone đi tiếp.

## 4. Công thức huấn luyện

Nguồn: sheet `Training`. Mỗi T01–T06 thay một trục so với T00; T07 kết hợp trục B và C. Chỉ một seed ở vòng sàng, nên chênh lệch nhỏ chưa đủ để khẳng định.

{training_table}

**Bổ sung phân tích:** yếu tố nào giúp, yếu tố nào không, và mức chênh lệch so với nhiễu qua seed.

## 5. Phương pháp suy luận

Nguồn: sheet `Inference`, `Latency`, `curves/inference_tradeoff.png`. Phương pháp chọn từ val: {method}.

{inference_table}

**Bổ sung phân tích:** ECE trước/sau hiệu chuẩn và đánh đổi F1 với p50/p95/p99.

## 6. Chung kết và lỗi dự đoán

Nguồn: sheet `Final`, `PerClass`, file dự đoán test và `curves/F01_confusion.png`.

{final_table}

Recall test Chinee Apple: {_format_mean_std(float(final['recall'][0][0]), float(final['recall'][1][0]))}; Snake Weed: {_format_mean_std(float(final['recall'][0][7]), float(final['recall'][1][7]))}. Cặp nhầm nhiều nhất tính trên tổng ma trận: {hard_pair_text}. Xem `misclassified_seed0.csv` và ảnh gốc để viết giả thuyết nguyên nhân.

## 7. Kết luận và khuyến nghị

Macro-F1 cải thiện {delta:+.4f} so với mốc. {comparison} **Bổ sung:** cấu hình đạt giới hạn p95 ≤ 100 ms nếu có, và thành phần đóng góp nhiều nhất theo bảng số liệu.

## 8. Hạn chế và việc tiếp theo

Chỉ một fold và ba seed ở chung kết. Fold được chia ngẫu nhiên, không theo địa điểm chụp, nên điểm test có thể lạc quan khi gặp ảnh khác miền. Ghi rõ các thí nghiệm chưa làm và lỗi phát hiện sau khi mở test, nếu có.

## 9. Phụ lục

Danh sách cấu hình, seed và log: xem `results.xlsx` và `runs/`. **Bổ sung link notebook Kaggle sau khi chạy.**
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    return {"final_macro_f1_mean": final["macro_f1"][0],
            "baseline_macro_f1_mean": base["macro_f1"][0], "delta": delta,
            "results": str(RESULTS), "report": str(ROOT / "report.md")}
