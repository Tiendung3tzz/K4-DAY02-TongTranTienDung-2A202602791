"""DeepWeeds data loading, split validation, transforms and DataLoaders."""
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
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
    "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read the authors' unchanged train/val/test CSVs for one fold."""
    labels_dir = Path(labels_dir)
    result = []
    for name in ("train", "val", "test"):
        path = labels_dir / f"{name}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {path}")
        frame = pd.read_csv(path)
        required = {"Filename", "Label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain {sorted(required)}")
        frame = frame.copy()
        frame["Filename"] = frame["Filename"].astype(str)
        frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
        result.append(frame)
    return tuple(result)


def check_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str | Path,
    fold: int = 0,
) -> dict:
    """Validate counts, labels, overlap, total size and image existence."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    per_class: dict[str, dict[int, int]] = {}

    for split, frame in frames.items():
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"Null Filename or Label found in {split}")
        if frame["Filename"].duplicated().any():
            duplicates = frame.loc[frame["Filename"].duplicated(), "Filename"].tolist()
            raise ValueError(f"Duplicate filenames in {split}: {duplicates[:10]}")

        split_names = set(frame["Filename"].astype(str))
        names[split] = split_names
        counts[split] = len(frame)
        label_values = pd.to_numeric(frame["Label"], errors="raise").astype(int)
        invalid = sorted(set(label_values) - set(range(NUM_CLASSES)))
        if invalid:
            raise ValueError(f"Invalid labels in {split}: {invalid}")
        per_class[split] = {
            int(k): int(v)
            for k, v in label_values.value_counts().sort_index().items()
        }

    overlap = {
        "train_val": names["train"] & names["val"],
        "train_test": names["train"] & names["test"],
        "val_test": names["val"] & names["test"],
    }
    nonempty = {key: sorted(value)[:10] for key, value in overlap.items() if value}
    if nonempty:
        raise ValueError(f"Split overlap is not empty: {nonempty}")

    union_count = len(set.union(*names.values()))
    expected_total = 17509
    if union_count != expected_total:
        raise ValueError(f"Fold {fold} union has {union_count} images, expected {expected_total}")

    expected_ratios = {"train": 0.60, "val": 0.20, "test": 0.20}
    ratios = {key: value / union_count for key, value in counts.items()}
    out_of_range = {
        key: ratio
        for key, ratio in ratios.items()
        if abs(ratio - expected_ratios[key]) > 0.01
    }
    if out_of_range:
        raise ValueError(f"Unexpected split ratios (more than 1 percentage point off): {out_of_range}")

    images_dir = Path(images_dir)
    all_names = set.union(*names.values())
    missing = [filename for filename in all_names if not (images_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files referenced by CSV are missing; examples: {missing[:10]}"
        )

    result = {
        "counts": counts,
        "ratios": ratios,
        "per_class": per_class,
        "overlap": {key: len(value) for key, value in overlap.items()},
        "union_count": union_count,
        "missing_images": len(missing),
    }
    print("Split audit:", result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic validation transforms or selected train augmentation."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")

    if train:
        operations = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ]
        if aug == "color":
            operations.append(T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ))
        elif aug == "trivial":
            operations.append(T.TrivialAugmentWide())
        elif aug == "randaug":
            operations.append(T.RandAugment())
        elif aug != "basic":
            raise ValueError(f"Unsupported augmentation: {aug}")
    else:
        # Resize the shorter side; this supports both the standard 224 crop and
        # resolution-sweep experiments without adding random validation transforms.
        operations = [
            T.Resize(max(256, img_size)),
            T.CenterCrop(img_size),
        ]

    operations.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T.Compose(operations)


class DeepWeedsDataset(Dataset):
    """Return (normalized image tensor, integer label, filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        required = {"Filename", "Label"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain {sorted(required)}")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        image_path = self.images_dir / filename
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    df: pd.DataFrame,
    images_dir: str | Path,
    transform,
    batch_size: int,
    train: bool,
    sampler: str | None = None,
    num_workers: int = 2,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    weighted_sampler = None
    if sampler == "balanced":
        labels = pd.to_numeric(df["Label"], errors="raise").astype(int)
        frequencies = labels.value_counts()
        sample_weights = labels.map(lambda value: 1.0 / frequencies[value])
        weighted_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    elif sampler is not None:
        raise ValueError("sampler must be None or 'balanced'")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train and weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )