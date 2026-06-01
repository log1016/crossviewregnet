#!/usr/bin/env python3
import argparse
import concurrent.futures as futures
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossviewregnet.io import index_point_clouds, read_las_xyz
from crossviewregnet.projection import points_to_images
from crossviewregnet.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Convert individual-tree LAS/LAZ files to CrossViewRegNet NPZ tensors.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw .las or .laz files.")
    parser.add_argument("--metadata-csv", required=True, help="CSV containing filename and tree height columns.")
    parser.add_argument("--output-dir", required=True, help="Directory where NPZ files will be written.")
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--height-column", default="tree_H")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-side-views", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def build_tasks(args):
    frame = pd.read_csv(args.metadata_csv)
    if args.filename_column not in frame.columns:
        raise ValueError(f"Missing filename column: {args.filename_column}")
    if args.height_column not in frame.columns:
        raise ValueError(f"Missing height column: {args.height_column}")

    point_cloud_index = index_point_clouds(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    missing = []
    for _, row in frame.iterrows():
        stem = Path(str(row[args.filename_column])).stem
        input_path = point_cloud_index.get(stem)
        if input_path is None:
            missing.append(stem)
            continue
        output_path = output_dir / f"{stem}.npz"
        tasks.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "height": float(row[args.height_column]),
                "resolution": args.resolution,
                "num_side_views": args.num_side_views,
                "max_points": args.max_points,
                "overwrite": args.overwrite,
            }
        )
    return tasks, missing


def process_one(task):
    output_path = Path(task["output_path"])
    if output_path.exists() and not task["overwrite"]:
        return {"status": "skipped", "input_path": task["input_path"], "output_path": str(output_path)}
    try:
        points = read_las_xyz(task["input_path"])
        images = points_to_images(
            points,
            resolution=task["resolution"],
            num_side_views=task["num_side_views"],
            max_points=task["max_points"],
        )
        np.savez_compressed(output_path, images=images, height=task["height"])
        return {"status": "success", "input_path": task["input_path"], "output_path": str(output_path)}
    except Exception:
        return {
            "status": "failed",
            "input_path": task["input_path"],
            "output_path": str(output_path),
            "error": traceback.format_exc(),
        }


def main():
    args = parse_args()
    start = time.perf_counter()
    tasks, missing = build_tasks(args)
    if not tasks:
        raise RuntimeError("No valid preprocessing tasks were created.")

    results = []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_one, tasks), total=len(tasks), desc="Preprocessing"):
            results.append(result)

    elapsed = time.perf_counter() - start
    summary = {
        "input_dir": str(args.input_dir),
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "resolution": args.resolution,
        "num_side_views": args.num_side_views,
        "tasks": len(tasks),
        "missing_files": len(missing),
        "success": sum(r["status"] == "success" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "seconds": elapsed,
        "seconds_per_tree": elapsed / max(len(tasks), 1),
    }
    summary_path = args.summary_json or str(Path(args.output_dir) / "preprocess_summary.json")
    save_json(summary, summary_path)
    print(summary)

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print("Failed files:")
        for item in failed[:10]:
            print(item["input_path"])


if __name__ == "__main__":
    main()
