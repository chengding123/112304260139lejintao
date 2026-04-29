import argparse
import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
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


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(image)

    x_src_start = max(0, -dx)
    x_src_end = IMAGE_SIZE - max(0, dx)
    y_src_start = max(0, -dy)
    y_src_end = IMAGE_SIZE - max(0, dy)

    x_dst_start = max(0, dx)
    x_dst_end = IMAGE_SIZE - max(0, -dx)
    y_dst_start = max(0, dy)
    y_dst_end = IMAGE_SIZE - max(0, -dy)

    shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = image[y_src_start:y_src_end, x_src_start:x_src_end]
    return shifted


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


class DigitDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray | None,
        mean: float,
        std: float,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.85:
                dx = int(np.random.randint(-2, 3))
                dy = int(np.random.randint(-2, 3))
                image = shift_image(image, dx, dy)
            if np.random.rand() < 0.25:
                noise = np.random.normal(loc=0.0, scale=0.03, size=image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0)

        image = (image - self.mean) / self.std
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        if self.labels is None:
            return image_tensor

        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return image_tensor, label_tensor


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
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
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

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
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta_shifts: list[tuple[int, int]]) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    for images in loader:
        images = images.to(device)
        logits_sum = None
        for dx, dy in tta_shifts:
            shifted = shift_tensor_batch(images, dx, dy)
            logits = model(shifted)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        batch_predictions = logits_sum.argmax(dim=1).cpu().numpy()
        predictions.append(batch_predictions)

    return np.concatenate(predictions)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_shifts: list[tuple[int, int]],
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []

    for images in loader:
        images = images.to(device)
        prob_sum = None
        for dx, dy in tta_shifts:
            shifted = shift_tensor_batch(images, dx, dy)
            probs = torch.softmax(model(shifted), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        batch_probabilities = (prob_sum / len(tta_shifts)).cpu().numpy()
        probabilities.append(batch_probabilities)

    return np.concatenate(probabilities, axis=0)


def save_submission(predictions: np.ndarray, output_path: Path) -> None:
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1, dtype=np.int64),
            "Label": predictions.astype(np.int64),
        }
    )
    submission.to_csv(output_path, index=False)


def make_json_safe(data: dict) -> dict:
    safe_data = {}
    for key, value in data.items():
        if isinstance(value, Path):
            safe_data[key] = str(value)
        else:
            safe_data[key] = value
    return safe_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN for Kaggle Digit Recognizer.")
    parser.add_argument("--data-zip", type=Path, default=Path("digit-recognizer.zip"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("best_digit_cnn.pt"))
    parser.add_argument("--metrics-path", type=Path, default=Path("training_metrics.json"))
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, labels, test_images = load_competition_data(args.data_zip)
    full_mean = float(train_images.mean() / 255.0)
    full_std = float(train_images.std() / 255.0 + 1e-6)

    train_idx = np.arange(len(labels))
    val_idx = np.array([], dtype=np.int64)

    if not args.full_train:
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=args.val_size,
            stratify=labels,
            random_state=args.seed,
        )

    train_dataset = DigitDataset(train_images[train_idx], labels[train_idx], mean=full_mean, std=full_std, augment=True)
    test_dataset = DigitDataset(test_images, labels=None, mean=full_mean, std=full_std, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if not args.full_train:
        val_dataset = DigitDataset(train_images[val_idx], labels[val_idx], mean=full_mean, std=full_std, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_acc = 0.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    if not args.predict_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            anneal_strategy="cos",
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            if args.full_train:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                    }
                )
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4%}"
                )
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mean": full_mean,
                        "std": full_std,
                        "best_val_acc": None,
                        "best_epoch": best_epoch,
                        "args": make_json_safe(vars(args)),
                    },
                    args.model_path,
                )
            else:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4%}"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "mean": full_mean,
                            "std": full_std,
                            "best_val_acc": best_val_acc,
                            "best_epoch": best_epoch,
                            "args": make_json_safe(vars(args)),
                        },
                        args.model_path,
                    )

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_best_val_acc = checkpoint.get("best_val_acc", best_val_acc)
    best_val_acc = None if checkpoint_best_val_acc is None else float(checkpoint_best_val_acc)
    best_epoch = int(checkpoint.get("best_epoch", best_epoch))
    full_mean = float(checkpoint.get("mean", full_mean))
    full_std = float(checkpoint.get("std", full_std))

    probabilities = predict_probabilities(
        model,
        test_loader,
        device,
        tta_shifts=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
    )
    predictions = probabilities.argmax(axis=1)
    save_submission(predictions, args.submission)

    metrics = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "mean": full_mean,
        "std": full_std,
        "history": history,
        "submission_path": str(args.submission),
        "model_path": str(args.model_path),
    }
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if best_val_acc is None:
        print(f"Finished full-data training at epoch {best_epoch}")
    else:
        print(f"Best validation accuracy: {best_val_acc:.4%} at epoch {best_epoch}")
    print(f"Saved model to: {args.model_path}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(f"Saved submission to: {args.submission}")


if __name__ == "__main__":
    main()
