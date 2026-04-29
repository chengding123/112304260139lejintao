import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import digit_recognizer_cnn as base_mod
import digit_recognizer_ensemble as wide_mod


class KaggleTrainDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


def emnist_fix(x: torch.Tensor) -> torch.Tensor:
    # EMNIST samples are stored transposed relative to MNIST-style orientation.
    x = torch.rot90(x, k=-1, dims=(1, 2))
    x = torch.flip(x, dims=(2,))
    return x


def build_external_datasets(root: Path):
    common_to_tensor = transforms.ToTensor()
    emnist_transform = transforms.Compose(
        [
            common_to_tensor,
            transforms.Lambda(emnist_fix),
        ]
    )
    usps_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            common_to_tensor,
        ]
    )

    emnist_train = datasets.EMNIST(root=str(root), split="digits", train=True, download=True, transform=emnist_transform)
    emnist_test = datasets.EMNIST(root=str(root), split="digits", train=False, download=True, transform=emnist_transform)
    parts = [emnist_train, emnist_test]

    try:
        usps_train = datasets.USPS(root=str(root), train=True, download=True, transform=usps_transform)
        usps_test = datasets.USPS(root=str(root), train=False, download=True, transform=usps_transform)
        parts.extend([usps_train, usps_test])
    except Exception as exc:
        print(f"Warning: USPS download unavailable, continuing with EMNIST only. Reason: {exc}")

    return ConcatDataset(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggressive external-data training for Digit Recognizer.")
    parser.add_argument("--data-zip", type=Path, default=Path("digit-recognizer.zip"))
    parser.add_argument("--external-root", type=Path, default=Path("external_digit_data"))
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--external-limit", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=0.0010)
    parser.add_argument("--finetune-lr", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--model-path", type=Path, default=Path("external_wide_model.pt"))
    parser.add_argument("--submission", type=Path, default=Path("external_wide_submission.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("external_wide_metrics.json"))
    args = parser.parse_args()

    base_mod.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, train_labels, test_images = base_mod.load_competition_data(args.data_zip)
    mean = float(train_images.mean() / 255.0)
    std = float(train_images.std() / 255.0 + 1e-6)

    external_dataset = build_external_datasets(args.external_root)
    if args.external_limit > 0 and args.external_limit < len(external_dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(external_dataset), generator=generator)[: args.external_limit].tolist()
        external_dataset = Subset(external_dataset, indices)
    kaggle_train_dataset = KaggleTrainDataset(train_images, train_labels)
    kaggle_test_dataset = wide_mod.DigitDataset(test_images, labels=None)

    external_loader = DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    kaggle_train_loader = DataLoader(
        kaggle_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    kaggle_test_loader = DataLoader(
        kaggle_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = wide_mod.DigitEnsembleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"pretrain": [], "finetune": []}

    if args.pretrain_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.pretrain_lr,
            epochs=args.pretrain_epochs,
            steps_per_epoch=len(external_loader),
            pct_start=0.15,
            anneal_strategy="cos",
        )

        for epoch in range(1, args.pretrain_epochs + 1):
            train_loss, train_acc = wide_mod.train_one_epoch(
                model, external_loader, criterion, optimizer, scheduler, device, mean, std
            )
            history["pretrain"].append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
            print(
                f"Pretrain {epoch:02d}/{args.pretrain_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4%}"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.finetune_lr,
        epochs=args.finetune_epochs,
        steps_per_epoch=len(kaggle_train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = wide_mod.train_one_epoch(
            model, kaggle_train_loader, criterion, optimizer, scheduler, device, mean, std
        )
        history["finetune"].append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        print(
            f"Finetune {epoch:02d}/{args.finetune_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4%}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        args.model_path,
    )

    probabilities = wide_mod.predict_probabilities(
        model,
        kaggle_test_loader,
        device,
        mean,
        std,
        tta_shifts=[(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
    )
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions}).to_csv(
        args.submission,
        index=False,
    )

    metrics = {
        "device": str(device),
        "external_root": str(args.external_root),
        "external_dataset_sizes": {
            "total": len(external_dataset),
            "kaggle_train": len(kaggle_train_dataset),
            "kaggle_test": len(kaggle_test_dataset),
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
