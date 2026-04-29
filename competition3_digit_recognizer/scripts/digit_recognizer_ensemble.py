import argparse
import json
import math
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 28
NUM_CLASSES = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_competition_data(zip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with ZipFile(zip_path) as zip_file:
        with zip_file.open("train.csv") as train_file:
            train_df = pd.read_csv(train_file)
        with zip_file.open("test.csv") as test_file:
            test_df = pd.read_csv(test_file)

    labels = train_df["label"].to_numpy(dtype=np.int64)
    train_images = (
        train_df.drop(columns=["label"]).to_numpy(dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    )
    test_images = test_df.to_numpy(dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    return train_images, labels, test_images


def shift_tensor_batch(batch: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return batch

    shifted = torch.roll(batch, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0

    if dx > 0:
        shifted[..., :, :dx] = 0
    elif dx < 0:
        shifted[..., :, dx:] = 0
    return shifted


def normalize_batch(batch: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (batch - mean) / std


def augment_batch(batch: torch.Tensor) -> torch.Tensor:
    batch = batch.clone()
    device = batch.device
    batch_size = batch.size(0)
    mask = torch.rand(batch_size, device=device) < 0.90

    if mask.any():
        selected = batch[mask]
        count = selected.size(0)

        angles = (torch.rand(count, device=device) * 24.0 - 12.0) * (math.pi / 180.0)
        scales = torch.empty(count, device=device).uniform_(0.93, 1.07)
        translate_x = torch.empty(count, device=device).uniform_(-0.10, 0.10)
        translate_y = torch.empty(count, device=device).uniform_(-0.10, 0.10)

        theta = torch.zeros(count, 2, 3, device=device, dtype=selected.dtype)
        theta[:, 0, 0] = torch.cos(angles) * scales
        theta[:, 0, 1] = -torch.sin(angles) * scales
        theta[:, 1, 0] = torch.sin(angles) * scales
        theta[:, 1, 1] = torch.cos(angles) * scales
        theta[:, 0, 2] = translate_x
        theta[:, 1, 2] = translate_y

        grid = F.affine_grid(theta, selected.size(), align_corners=False)
        transformed = F.grid_sample(selected, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        batch[mask] = transformed

    noise_mask = torch.rand(batch_size, device=device) < 0.30
    if noise_mask.any():
        noise = torch.randn_like(batch[noise_mask]) * 0.025
        batch[noise_mask] = torch.clamp(batch[noise_mask] + noise, 0.0, 1.0)

    return batch


class DigitDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray | None) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0).unsqueeze(0)
        if self.labels is None:
            return image
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DigitEnsembleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.10),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images = normalize_batch(augment_batch(images), mean, std)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = normalize_batch(images.to(device), mean, std)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    tta_shifts: list[tuple[int, int]],
) -> np.ndarray:
    model.eval()
    all_probabilities: list[np.ndarray] = []

    for images in loader:
        images = images.to(device)
        probability_sum = None

        for dx, dy in tta_shifts:
            shifted = shift_tensor_batch(images, dx, dy)
            normalized = normalize_batch(shifted, mean, std)
            probabilities = torch.softmax(model(normalized), dim=1)
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities

        averaged_probabilities = (probability_sum / len(tta_shifts)).cpu().numpy()
        all_probabilities.append(averaged_probabilities)

    return np.concatenate(all_probabilities, axis=0)


def save_submission(probabilities: np.ndarray, output_path: Path) -> None:
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1, dtype=np.int64),
            "Label": predictions,
        }
    )
    submission.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a K-fold ensemble for Kaggle Digit Recognizer.")
    parser.add_argument("--data-zip", type=Path, default=Path("digit-recognizer.zip"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--submission", type=Path, default=Path("submission_ensemble.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("ensemble_metrics.json"))
    parser.add_argument("--models-dir", type=Path, default=Path("ensemble_models"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, labels, test_images = load_competition_data(args.data_zip)
    mean = float(train_images.mean() / 255.0)
    std = float(train_images.std() / 255.0 + 1e-6)

    args.models_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = DigitDataset(test_images, labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    fold_metrics: list[dict[str, float | int]] = []
    ensemble_probabilities = np.zeros((len(test_images), NUM_CLASSES), dtype=np.float32)

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(train_images, labels), start=1):
        print(f"\n=== Fold {fold_index}/{args.folds} ===")
        train_dataset = DigitDataset(train_images[train_idx], labels[train_idx])
        val_dataset = DigitDataset(train_images[val_idx], labels[val_idx])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = DigitEnsembleCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
        )

        best_val_acc = 0.0
        best_epoch = 0
        checkpoint_path = args.models_dir / f"fold_{fold_index}.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, mean, std
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, mean, std)

            print(
                f"Fold {fold_index} | Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4%}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "best_val_acc": best_val_acc,
                        "best_epoch": best_epoch,
                    },
                    checkpoint_path,
                )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

        fold_probabilities = predict_probabilities(
            model,
            test_loader,
            device,
            mean,
            std,
            tta_shifts=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)],
        )
        ensemble_probabilities += fold_probabilities / args.folds

        fold_metric = {
            "fold": fold_index,
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_val_acc": float(checkpoint["best_val_acc"]),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "checkpoint": str(checkpoint_path),
        }
        fold_metrics.append(fold_metric)
        print(
            f"Fold {fold_index} best validation accuracy: "
            f"{fold_metric['best_val_acc']:.4%} at epoch {fold_metric['best_epoch']}"
        )

    save_submission(ensemble_probabilities, args.submission)

    mean_val_acc = float(np.mean([fold["best_val_acc"] for fold in fold_metrics]))
    metrics = {
        "device": str(device),
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "mean": mean,
        "std": std,
        "mean_val_acc": mean_val_acc,
        "fold_metrics": fold_metrics,
        "submission_path": str(args.submission),
    }
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nEnsemble mean validation accuracy: {mean_val_acc:.4%}")
    print(f"Saved submission to: {args.submission}")
    print(f"Saved metrics to: {args.metrics_path}")


if __name__ == "__main__":
    main()
