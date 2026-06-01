#!/usr/bin/env python3
import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossviewregnet.dataset import TreeNPZDataset
from crossviewregnet.model import CrossViewRegNet
from crossviewregnet.utils import (
    build_eval_transform,
    build_train_transform,
    classification_metrics,
    height_stats,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrossViewRegNet.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--data-dir", required=True, help="Directory containing preprocessed training/validation NPZ files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--num-views", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--height-noise", type=float, default=0.01)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", default="species_id")
    parser.add_argument("--height-column", default="tree_H")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def make_loaders(args, height_mean, height_std):
    train_dataset = TreeNPZDataset(
        args.train_csv,
        args.data_dir,
        transform=build_train_transform(),
        filename_column=args.filename_column,
        label_column=args.label_column,
        height_column=args.height_column,
        height_mean=height_mean,
        height_std=height_std,
        height_noise=args.height_noise,
        require_labels=True,
    )
    val_dataset = TreeNPZDataset(
        args.val_csv,
        args.data_dir,
        transform=build_eval_transform(),
        filename_column=args.filename_column,
        label_column=args.label_column,
        height_column=args.height_column,
        height_mean=height_mean,
        height_std=height_std,
        height_noise=0.0,
        require_labels=True,
    )

    weights = train_dataset.sampler_weights()
    if weights is not None and args.samples_per_epoch > 0:
        sampler = torch.utils.data.WeightedRandomSampler(weights, args.samples_per_epoch, replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, heights, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        heights = heights.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            logits = model(images, heights)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for images, heights, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            heights = heights.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images, heights)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    metrics = classification_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_loss_curve(metrics, output_dir):
    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = Path(output_dir) / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_mean, height_std = height_stats(args.train_csv, args.height_column)
    train_loader, val_loader = make_loaders(args, height_mean, height_std)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossViewRegNet(
        n_classes=args.num_classes,
        n_views=args.num_views,
        pretrained_backbone=not args.no_pretrained,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    if args.amp and device == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = output_dir / f"best_model_{timestamp}.pth"
    metrics_path = output_dir / "metrics.csv"

    best_acc = -1.0
    epochs_without_improvement = 0
    metrics_rows = []

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / max(args.warmup_epochs, 1)
            for group in optimizer.param_groups:
                group["lr"] = lr

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch > args.warmup_epochs:
            scheduler.step(val_metrics["accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(row)

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    loss_curve = save_loss_curve(metrics_rows, output_dir)
    summary = {
        "best_model_path": str(best_path),
        "metrics_path": str(metrics_path),
        "loss_curve": loss_curve,
        "best_val_accuracy": best_acc,
        "epochs_completed": len(metrics_rows),
        "height_mean": height_mean,
        "height_std": height_std,
        "device": device,
        "args": vars(args),
    }
    save_json(summary, output_dir / "training_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
