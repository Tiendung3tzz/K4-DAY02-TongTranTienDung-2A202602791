"""DeepWeeds split, transforms and data loaders. CSV rows are never modified."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
               "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    if fold not in range(5):
        raise ValueError("fold must be between 0 and 4")
    root = Path(labels_dir)
    result = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_subset{fold}.csv"
        frame = pd.read_csv(path)
        # Official fold CSVs have only Filename and Label; Species is in labels.csv.
        if not {"Filename", "Label"} <= set(frame.columns):
            raise ValueError(f"{path}: missing Filename or Label (found {list(frame.columns)})")
        result.append(frame)
    return tuple(result)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame,
                test_df: pd.DataFrame, images_dir: str | Path) -> dict:
    frames = {"train": train_df, "val": val_df, "test": test_df}
    sets = {}
    per_class = {}
    root = Path(images_dir)
    for split, frame in frames.items():
        if frame.empty or frame[["Filename", "Label"]].isna().any().any():
            raise ValueError(f"{split}: empty rows or missing Filename/Label")
        names = frame["Filename"].astype(str)
        if names.duplicated().any():
            raise ValueError(f"{split}: duplicate Filename")
        labels = pd.to_numeric(frame["Label"], errors="raise")
        if not labels.isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{split}: labels must be integers 0..8")
        sets[split] = set(names)
        per_class[split] = {int(k): int(v) for k, v in
                            labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0).items()}
        if any(count == 0 for count in per_class[split].values()):
            raise ValueError(f"{split}: at least one of the nine classes is absent")
        missing = [name for name in names if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} images missing; first: {missing[:5]}")
    overlaps = {"train_val": len(sets["train"] & sets["val"]),
                "train_test": len(sets["train"] & sets["test"]),
                "val_test": len(sets["val"] & sets["test"])}
    if any(overlaps.values()):
        raise ValueError(f"Split overlap: {overlaps}")
    total = len(set.union(*sets.values()))
    if total != 17509:
        raise ValueError(f"Expected 17,509 unique images, found {total}")
    counts = {key: len(frame) for key, frame in frames.items()}
    expected = {"train": 0.6, "val": 0.2, "test": 0.2}
    for split in frames:
        if abs(counts[split] / total - expected[split]) > 0.01:
            raise ValueError(f"{split}: count differs by over 1 percentage point from expected split")
    report = {"n": counts, "per_class": per_class, "overlap": overlaps, "total": total}
    print(report)
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if img_size < 1:
        raise ValueError("img_size must be positive")
    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError(f"Unknown augmentation: {aug}")
    if train:
        steps = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
        if aug == "color":
            steps.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            steps.append(T.TrivialAugmentWide())
        elif aug == "randaug":
            steps.append(T.RandAugment())
    else:
        steps = [T.Resize(256), T.CenterCrop(img_size)]
    return T.Compose([*steps, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else T.ToTensor()(image)
        return tensor, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    if sampler not in {None, "balanced"} or (not train and sampler is not None):
        raise ValueError("balanced sampler is available only for train")
    ds = DeepWeedsDataset(df, images_dir, transform)
    weighted = None
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = torch.as_tensor([1.0 / counts[int(label)] for label in df["Label"]], dtype=torch.double)
        weighted = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and weighted is None,
                      sampler=weighted, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) >= batch_size,
                      worker_init_fn=_seed_worker if num_workers else None)
