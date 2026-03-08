#!/usr/bin/env python3
"""
Count visual tokens across all .pt files in a pointer_memory folder.

Usage:
  python scripts/demo/count_tokens_folder.py data/media/robofac/pointer_memory
  python scripts/demo/count_tokens_folder.py data/media/robofac/pointer_memory --workers 16
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _load_pt(path):
    import torch
    data = torch.load(path, weights_only=True)
    ts = data.get("pointer_timestamps")
    if ts is None:
        return None
    return (ts.shape[0], ((ts.max().item() + 1) + 1) // 2)


def main():
    parser = argparse.ArgumentParser(description="Count visual tokens in a pointer_memory folder")
    parser.add_argument("folder", help="Path to pointer_memory folder containing .pt files")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from tqdm import tqdm

    pt_files = sorted(Path(args.folder).rglob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {args.folder}")

    if not pt_files:
        return

    token_counts = []
    frame_counts = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_load_pt, str(p)): p for p in pt_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading"):
            result = fut.result()
            if result is None:
                continue
            token_counts.append(result[0])
            frame_counts.append(result[1])

    n = len(token_counts)
    avg_tok = sum(token_counts) / n
    avg_frm = sum(frame_counts) / n
    print(f"\nFiles loaded : {n}")
    print(f"Avg tokens   : {avg_tok:.1f}")
    print(f"Min tokens   : {min(token_counts)}")
    print(f"Max tokens   : {max(token_counts)}")
    print(f"Avg frames   : {avg_frm:.1f}")


if __name__ == "__main__":
    main()
