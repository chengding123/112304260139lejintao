import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import digit_recognizer_cnn as base_mod
import digit_recognizer_ensemble as ens_mod


class WeightedDigitDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, weights: np.ndarray, mean: float, std: float) -> None:
        self.images = images
        self.labels = labels
        self.weights = weights
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32) / 255.0

        if np.random.rand() < 0.90:
            dx = int(np.random.randint(-2, 3))
            dy = int(np.random.randint(-2, 3))
            image = base_mod.shift_image(image, dx, dy)

        if np.random.rand() < 0.50:
            image_tensor = torch.from_numpy(image).unsqueeze(0)
            angle = float(np.random.uniform(-14.0, 14.0))
            scale = float(np.random.uniform(0.93, 1.07))
            tx = float(np.random.uniform(-0.10, 0.10))
            ty = float(np.random.uniform(-0.10, 0.10))
            angle_rad = angle * np.pi / 180.0
            theta = torch.tensor(
                [
                    [np.cos(angle_rad) * scale, -np.sin(angle_rad) * scale, tx],
                    [np.sin(angle_rad) * scale, np.cos(angle_rad) * scale, ty],
                ],
                dtype=torch.float32,
            ).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, size=(1, 1, 28, 28), align_corners=False)
            image = (
                torch.nn.functional.grid_sample(
                    image_tensor.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )

        if np.random.rand() < 0.30:
            noise = np.random.normal(loc=0.0, scale=0.02, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        image = (image - self.mean) / self.std
        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )


def build_pseudo_labels(root: Path, device: torch.device, test_images: np.ndarray, mean: float, std: float):
    shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    base_dataset = base_mod.DigitDataset(test_images, labels=None, mean=mean, std=std, augment=False)
    base_loader = DataLoader(base_dataset, batch_size=256, shuffle=False, num_workers=0)

    ens_dataset = ens_mod.DigitDataset(test_images, labels=None)
    ens_loader = DataLoader(ens_dataset, batch_size=256, shuffle=False, num_workers=0)

    members = []
    base_members = [
        "full_digit_cnn_seed2026.pt",
        "full_digit_cnn_seed3407.pt",
        "full_digit_cnn_seed7.pt",
        "best_digit_cnn.pt",
    ]
    for name in base_members:
        checkpoint = torch.load(root / name, map_location=device, weights_only=False)
        model = base_mod.DigitCNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        probabilities = base_mod.predict_probabilities(model, base_loader, device, shifts)
        members.append(probabilities)

    ensemble_members = ["ensemble_models/fold_1.pt", "ensemble_models/fold_2.pt"]
    for name in ensemble_members:
        checkpoint = torch.load(root / name, map_location=device, weights_only=False)
        model = ens_mod.DigitEnsembleCNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        probabilities = ens_mod.predict_probabilities(model, ens_loader, device, mean, std, shifts)
        members.append(probabilities)

    probability_stack = np.stack(members, axis=0)
    hard_predictions = probability_stack.argmax(axis=2).transpose(1, 0)

    pseudo_labels = []
    pseudo_weights = []
    pseudo_indices = []

    for idx, row in enumerate(hard_predictions):
        labels, counts = np.unique(row, return_counts=True)
        best_idx = counts.argmax()
        majority_count = int(counts[best_idx])
        majority_label = int(labels[best_idx])

        if majority_count >= 5:
            pseudo_indices.append(idx)
            pseudo_labels.append(majority_label)
            pseudo_weights.append(0.35)
        elif majority_count == 4:
            pseudo_indices.append(idx)
            pseudo_labels.append(majority_label)
            pseudo_weights.append(0.20)

    return (
        np.array(pseudo_indices, dtype=np.int64),
        np.array(pseudo_labels, dtype=np.int64),
        np.array(pseudo_weights, dtype=np.float32),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, weights in loader:
        images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pseudo-label digit recognizer model.")
    parser.add_argument("--data-zip", type=Path, default=Path("digit-recognizer.zip"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--model-path", type=Path, default=Path("pseudo_digit_cnn.pt"))
    parser.add_argument("--submission", type=Path, default=Path("pseudo_submission.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("pseudo_metrics.json"))
    args = parser.parse_args()

    base_mod.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, train_labels, test_images = base_mod.load_competition_data(args.data_zip)
    mean = float(train_images.mean() / 255.0)
    std = float(train_images.std() / 255.0 + 1e-6)

    pseudo_indices, pseudo_labels, pseudo_weights = build_pseudo_labels(Path("."), device, test_images, mean, std)
    print(f"Pseudo-labeled samples: {len(pseudo_indices)}")

    mixed_images = np.concatenate([train_images, test_images[pseudo_indices]], axis=0)
    mixed_labels = np.concatenate([train_labels, pseudo_labels], axis=0)
    mixed_weights = np.concatenate(
        [np.ones(len(train_images), dtype=np.float32), pseudo_weights.astype(np.float32)],
        axis=0,
    )

    train_dataset = WeightedDigitDataset(mixed_images, mixed_labels, mixed_weights, mean=mean, std=std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_dataset = base_mod.DigitDataset(test_images, labels=None, mean=mean, std=std, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = base_mod.DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4%}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "pseudo_count": int(len(pseudo_indices)),
        },
        args.model_path,
    )

    probabilities = base_mod.predict_probabilities(
        model,
        test_loader,
        device,
        tta_shifts=[(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
    )
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions}).to_csv(
        args.submission, index=False
    )

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "pseudo_count": int(len(pseudo_indices)),
        "pseudo_weight_stats": {
            "min": float(pseudo_weights.min()) if len(pseudo_weights) else None,
            "max": float(pseudo_weights.max()) if len(pseudo_weights) else None,
            "mean": float(pseudo_weights.mean()) if len(pseudo_weights) else None,
        },
        "history": history,
        "model_path": str(args.model_path),
        "submission_path": str(args.submission),
    }
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {args.model_path}")
    print(f"Saved submission to: {args.submission}")
    print(f"Saved metrics to: {args.metrics_path}")


if __name__ == "__main__":
    main()
