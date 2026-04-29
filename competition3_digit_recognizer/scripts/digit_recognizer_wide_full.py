import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import digit_recognizer_cnn as base_mod
import digit_recognizer_ensemble as wide_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a full-data wide CNN for Digit Recognizer.")
    parser.add_argument("--data-zip", type=Path, default=Path("digit-recognizer.zip"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--model-path", type=Path, default=Path("wide_full_model.pt"))
    parser.add_argument("--submission", type=Path, default=Path("wide_full_submission.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("wide_full_metrics.json"))
    args = parser.parse_args()

    base_mod.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, train_labels, test_images = base_mod.load_competition_data(args.data_zip)
    mean = float(train_images.mean() / 255.0)
    std = float(train_images.std() / 255.0 + 1e-6)

    train_dataset = wide_mod.DigitDataset(train_images, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_dataset = wide_mod.DigitDataset(test_images, labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = wide_mod.DigitEnsembleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        train_loss, train_acc = wide_mod.train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, mean, std
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4%}")

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
        test_loader,
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "submission_path": str(args.submission),
        "history": history,
    }
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {args.model_path}")
    print(f"Saved submission to: {args.submission}")
    print(f"Saved metrics to: {args.metrics_path}")


if __name__ == "__main__":
    main()
