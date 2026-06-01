#!/usr/bin/env python3
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossviewregnet.dataset import TreeNPZDataset
from crossviewregnet.model import CrossViewRegNet
from crossviewregnet.utils import build_eval_transform, build_train_transform, height_stats, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run CrossViewRegNet inference.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lookup-csv", default=None)
    parser.add_argument("--train-csv", default=None, help="Training CSV used to compute height normalization statistics.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--num-views", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--height-column", default="tree_H")
    parser.add_argument("--submission-format", action="store_true")
    parser.add_argument("--debug-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def tree_id_from_filename(value):
    stem = Path(str(value)).stem
    match = re.search(r"\d+", stem)
    return int(match.group(0)) if match else stem


def main():
    args = parse_args()
    if args.train_csv:
        height_mean, height_std = height_stats(args.train_csv, args.height_column)
    else:
        frame = pd.read_csv(args.test_csv)
        height_mean = float(frame[args.height_column].mean()) if args.height_column in frame.columns else 0.0
        height_std = float(frame[args.height_column].std()) if args.height_column in frame.columns else 1.0

    transform = build_train_transform() if args.tta > 1 else build_eval_transform()
    dataset = TreeNPZDataset(
        args.test_csv,
        args.data_dir,
        transform=transform,
        filename_column=args.filename_column,
        height_column=args.height_column,
        height_mean=height_mean,
        height_std=height_std,
        height_noise=0.01 if args.tta > 1 else 0.0,
        require_labels=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossViewRegNet(n_classes=args.num_classes, n_views=args.num_views, pretrained_backbone=False)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    identifiers = dataset.frame[args.filename_column].astype(str).tolist()
    probs_by_identifier = {identifier: None for identifier in identifiers}

    start = time.perf_counter()
    with torch.no_grad():
        for tta_idx in range(args.tta):
            for images, heights, batch_ids in tqdm(loader, desc=f"TTA {tta_idx + 1}/{args.tta}"):
                images = images.to(device, non_blocking=True)
                heights = heights.to(device, non_blocking=True)
                logits = model(images, heights)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for idx, identifier in enumerate(batch_ids):
                    if probs_by_identifier[identifier] is None:
                        probs_by_identifier[identifier] = probs[idx]
                    else:
                        probs_by_identifier[identifier] += probs[idx]

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pred_ids = {key: int(np.argmax(value)) for key, value in probs_by_identifier.items()}
    species_lookup = None
    if args.lookup_csv:
        lookup = pd.read_csv(args.lookup_csv)
        species_lookup = dict(zip(lookup["species_id"].astype(int), lookup["species"].astype(str)))

    rows = []
    debug_rows = []
    for identifier in identifiers:
        species_id = pred_ids[identifier]
        species_name = species_lookup.get(species_id, str(species_id)) if species_lookup else str(species_id)
        if args.submission_format:
            rows.append({"treeID": tree_id_from_filename(identifier), "predicted_species": species_name})
        else:
            rows.append({"filename": identifier, "species_id": species_id, "predicted_species": species_name})
        debug_rows.append(
            {
                "filename": identifier,
                "treeID": tree_id_from_filename(identifier),
                "species_id": species_id,
                "predicted_species": species_name,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    debug_csv = args.debug_csv or str(output_csv.with_name(output_csv.stem + "_debug.csv"))
    pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "data_dir": args.data_dir,
        "output_csv": str(output_csv),
        "debug_csv": debug_csv,
        "samples": len(dataset),
        "tta": args.tta,
        "batch_size": args.batch_size,
        "prediction_seconds": elapsed,
    }
    save_json(summary, args.summary_json or str(output_csv.with_suffix(".summary.json")))
    print(summary)


if __name__ == "__main__":
    main()
