# core/dataset.py
#
# Dataset loading for CIFAR-10, CIFAR-100, and TinyImageNet.
#
# All three datasets are returned as 32x32 RGB images so architectures
# never need to know which dataset they are processing.  TinyImageNet
# ships at 64x64; we downsample with Resize(36) + CenterCrop(32).
#
# Normalization stats are per-dataset to keep activations in a healthy
# range regardless of which dataset is used.
#
# Eval subset
# -----------
# collect_eval() returns at most n samples from the eval loader.
# The slice is always the first n — not random — so the same n samples
# are used for kappa measurement in every experiment that uses the same
# dataset, making kappa values directly comparable across conditions.
#
# TinyImageNet preprocessing
# ---------------------------
# Training:   Resize(40) + RandomCrop(32) + HFlip  (standard downscale augmentation)
# Evaluation: Resize(36) + CenterCrop(32)           (deterministic center crop)
# Both produce 32x32 images matching CIFAR spatial resolution.
#
# The val/ directory must be restructured before use: the Stanford distribution
# stores all validation images flat in val/images/ with labels in
# val/val_annotations.txt, but ImageFolder expects val/{class_id}/*.JPEG.
# TinyImageNet val/ must be restructured before use; see the download note in get_dataset().

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple


# per-dataset channel mean and std
_STATS = {
    "cifar10":      ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100":     ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}

N_CLASSES = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}


def _train_transform(name: str) -> transforms.Compose:
    mean, std = _STATS[name]
    if name == "tinyimagenet":
        # TinyImageNet ships at 64x64.  Resize to 40 then random-crop to 32
        # gives the same spatial coverage as the CIFAR augmentation while
        # keeping every image at 32x32 — matching the eval transform.
        return transforms.Compose([
            transforms.Resize(40),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # CIFAR-10/100 are already 32x32; standard augmentation.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _eval_transform(name: str) -> transforms.Compose:
    mean, std = _STATS[name]
    if name == "tinyimagenet":
        # TinyImageNet is 64x64; resize+centercrop to 32x32.
        return transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # CIFAR-10/100 are already 32x32 — standard eval: just normalise.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_dataset(name: str, data_root: str, train: bool):
    """Return a torchvision Dataset for name in {cifar10, cifar100, tinyimagenet}.

    Applies training or evaluation transforms automatically.
    TinyImageNet val/ must be restructured so each class has its own subdirectory.
    """
    name = name.lower()
    os.makedirs(data_root, exist_ok=True)
    tf = _train_transform(name) if train else _eval_transform(name)

    if name == "cifar10":
        return datasets.CIFAR10(data_root, train=train, download=True, transform=tf)
    elif name == "cifar100":
        return datasets.CIFAR100(data_root, train=train, download=True, transform=tf)
    elif name == "tinyimagenet":
        split = "train" if train else "val"
        path  = os.path.join(data_root, "tiny-imagenet-200", split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TinyImageNet not found at {path}.\n"
                "Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                f"Extract to {data_root}/tiny-imagenet-200/ and restructure val/ "
                "so each class label has its own subdirectory."
            )
        return datasets.ImageFolder(path, transform=tf)
    else:
        raise ValueError(f"unknown dataset '{name}'; valid: cifar10, cifar100, tinyimagenet")


def get_n_classes(name: str) -> int:
    """Return number of classes for a dataset (10 / 100 / 200)."""
    return N_CLASSES[name.lower()]


def get_loaders(
    name:       str,
    data_root:  str = "./data",
    batch_size: int = 256,
    n_workers:  int = 4,
    n_eval:     int = 8192,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, eval_loader).
    eval_loader is capped at n_eval samples taken from the front of the
    eval split, so kappa measurements are always over the same images.
    """
    train_ds = get_dataset(name, data_root, train=True)
    eval_ds  = get_dataset(name, data_root, train=False)

    n       = min(n_eval, len(eval_ds))
    eval_ds = Subset(eval_ds, list(range(n)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=pin, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=pin,
    )
    return train_loader, eval_loader


def collect_eval(
    eval_loader: DataLoader,
    n:           int = 8192,
    seed:        int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pull at most n samples from eval_loader and shuffle with a fixed seed.

    Shuffling ensures all classes are represented even when the underlying
    dataset is sorted by class (e.g. TinyImageNet val/ via ImageFolder).
    Without shuffling, a sorted dataset yields only the first ~82 of 200
    classes in 4096 samples, making kappa_task incomparable across datasets.

    Returns
    -------
    X_raw  [N, 3, 32, 32]  raw image tensors (passed to model.forward)
    X_flat [N, 3072]       flattened inputs   (passed to MI estimator as X)
    Y      [N]             integer class labels
    """
    xs, ys = [], []
    total  = 0
    for xb, yb in eval_loader:
        xs.append(xb)
        ys.append(yb)
        total += xb.shape[0]
        if total >= n:
            break

    X_raw = torch.cat(xs)[:n]
    Y     = torch.cat(ys)[:n]

    # shuffle with a fixed seed so the same n samples are always selected
    # in the same order — reproducible across conditions
    g   = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(X_raw.shape[0], generator=g)
    X_raw  = X_raw[idx]
    Y      = Y[idx]

    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    return X_raw, X_flat, Y
