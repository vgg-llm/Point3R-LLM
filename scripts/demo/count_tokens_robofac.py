#!/usr/bin/env python3
"""
Count visual (pointer) tokens per video from a RoboFAC evaluation JSONL.

For each unique pointer_data .pt file, loads it and reports:
  - Token count:  pointer_timestamps.shape[0]
  - Estimated sampled frames (non-Point3R baseline):
        ((timestamps.max() + 1) + 1) // 2

Statistics are grouped by experiment type:
  Short-horizon, Medium-horizon, Long-horizon, Dynamic, Real-world.

Usage:
  python scripts/demo/count_tokens_robofac.py \
    --jsonl_path logs/.../samples_robofac_point3r.jsonl

  # Custom base dir for pointer_data resolution:
  python scripts/demo/count_tokens_robofac.py \
    --jsonl_path <path> --base_dir data/media
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Experiment-type partitioning (matching robofac_point3r/utils.py)
EXPERIMENT_TYPES = {
    "Short-horizon": ["PickCube", "PullCube", "PushCube", "StackCube", "LiftPegUpright"],
    "Medium-horizon": ["PlugCharger", "PegInsertionSide", "UprightStack", "PullCubeTool",
                       "InsertCylinder", "PlaceCube", "ToolsTask"],
    "Long-horizon": ["SafeTask", "MicrowaveTask"],
    "Dynamic": ["SpinStack", "SpinPullStack"],
}
TASK_TO_EXPERIMENT_TYPE = {
    task: exp_type
    for exp_type, tasks in EXPERIMENT_TYPES.items()
    for task in tasks
}

EXP_TYPE_ORDER = ["Short-horizon", "Medium-horizon", "Long-horizon", "Dynamic", "Real-world"]


def _get_experiment_type(doc):
    if doc.get("data_source") == "realworld":
        return "Real-world"
    return TASK_TO_EXPERIMENT_TYPE.get(doc.get("task", ""), "Unknown")


def _load_pt(args_tuple):
    """Worker: load a single .pt file and return (pd_rel, token_count, estimated_frames) or None."""
    import torch
    pd_rel, full_path = args_tuple
    if not os.path.isfile(full_path):
        return (pd_rel, None, None, "missing")
    data = torch.load(full_path, weights_only=True)
    timestamps = data.get("pointer_timestamps")
    if timestamps is None:
        return (pd_rel, None, None, "no_timestamps")
    token_count = timestamps.shape[0]
    estimated_frames = ((timestamps.max().item() + 1) + 1) // 2
    return (pd_rel, token_count, estimated_frames, None)


def main():
    parser = argparse.ArgumentParser(description="Count visual tokens per video in RoboFAC JSONL")
    parser.add_argument("--jsonl_path", required=True,
                        help="Path to samples JSONL from lmms_eval")
    parser.add_argument("--base_dir", default="data/media",
                        help="Base directory for resolving pointer_data paths (default: data/media)")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON path for detailed results")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for loading .pt files (default: 8)")
    args = parser.parse_args()

    from tqdm import tqdm

    print("=" * 70)
    print("Visual Token Statistics for RoboFAC Evaluation")
    print("=" * 70)
    print()
    print("This script loads pre-computed pointer_data (.pt) files referenced")
    print("in a RoboFAC evaluation JSONL and reports two metrics per video:")
    print()
    print("  Avg Tokens         – Number of Point3R visual tokens injected into")
    print("                       the LLM (pointer_timestamps.shape[0]).")
    print("  Avg Sampled Frames – Estimated video frames a non-Point3R baseline")
    print("                       would sample: ((max_timestamp + 1) + 1) // 2.")
    print()
    print(f"  JSONL  : {args.jsonl_path}")
    print(f"  Base   : {args.base_dir}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)
    print()

    # --- Read JSONL ---
    docs = []
    with open(args.jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line)["doc"])
    print(f"Loaded {len(docs)} samples from {args.jsonl_path}")

    # --- Deduplicate by pointer_data (each .pt file = one video) ---
    # Map pointer_data path -> first doc (for experiment type lookup)
    pointer_to_doc = {}
    for doc in docs:
        pd_path = doc.get("pointer_data", "")
        if pd_path and pd_path not in pointer_to_doc:
            pointer_to_doc[pd_path] = doc

    unique_videos = list(pointer_to_doc.keys())
    print(f"Unique videos (pointer_data files): {len(unique_videos)}")

    # --- Load .pt files in parallel and extract stats ---
    work_items = [(pd_rel, os.path.join(args.base_dir, pd_rel)) for pd_rel in unique_videos]

    stats_by_type = defaultdict(list)
    missing = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_load_pt, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading .pt files"):
            pd_rel, token_count, estimated_frames, err = future.result()
            if err == "missing":
                missing += 1
                if missing <= 5:
                    print(f"  WARNING: missing {os.path.join(args.base_dir, pd_rel)}")
                continue
            if err == "no_timestamps":
                print(f"  WARNING: no pointer_timestamps in {os.path.join(args.base_dir, pd_rel)}")
                continue
            exp_type = _get_experiment_type(pointer_to_doc[pd_rel])
            stats_by_type[exp_type].append((token_count, estimated_frames))

    if missing:
        print(f"  ({missing} total missing .pt files)")

    # --- Print summary table ---
    header = f"{'Experiment Type':<20} | {'# Videos':>8} | {'Avg Tokens':>10} | {'Avg Sampled Frames':>18}"
    sep = "-" * 20 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 18

    print(f"\n{header}")
    print(sep)

    all_tokens = []
    all_frames = []
    output_data = {}

    for exp_type in EXP_TYPE_ORDER:
        entries = stats_by_type.get(exp_type, [])
        if not entries:
            print(f"{exp_type:<20} | {'N/A':>8} | {'N/A':>10} | {'N/A':>18}")
            continue
        tokens = [e[0] for e in entries]
        frames = [e[1] for e in entries]
        avg_tok = sum(tokens) / len(tokens)
        avg_frm = sum(frames) / len(frames)
        print(f"{exp_type:<20} | {len(entries):>8} | {avg_tok:>10.1f} | {avg_frm:>18.1f}")
        all_tokens.extend(tokens)
        all_frames.extend(frames)
        output_data[exp_type] = {
            "num_videos": len(entries),
            "avg_tokens": round(avg_tok, 2),
            "avg_sampled_frames": round(avg_frm, 2),
        }

    # Handle unknown types
    for exp_type, entries in stats_by_type.items():
        if exp_type not in EXP_TYPE_ORDER:
            tokens = [e[0] for e in entries]
            frames = [e[1] for e in entries]
            avg_tok = sum(tokens) / len(tokens)
            avg_frm = sum(frames) / len(frames)
            print(f"{exp_type:<20} | {len(entries):>8} | {avg_tok:>10.1f} | {avg_frm:>18.1f}")
            all_tokens.extend(tokens)
            all_frames.extend(frames)
            output_data[exp_type] = {
                "num_videos": len(entries),
                "avg_tokens": round(avg_tok, 2),
                "avg_sampled_frames": round(avg_frm, 2),
            }

    print(sep)
    if all_tokens:
        overall_avg_tok = sum(all_tokens) / len(all_tokens)
        overall_avg_frm = sum(all_frames) / len(all_frames)
        print(f"{'Overall':<20} | {len(all_tokens):>8} | {overall_avg_tok:>10.1f} | {overall_avg_frm:>18.1f}")
        output_data["Overall"] = {
            "num_videos": len(all_tokens),
            "avg_tokens": round(overall_avg_tok, 2),
            "avg_sampled_frames": round(overall_avg_frm, 2),
        }
    else:
        print("No data loaded.")

    # --- Save JSON output ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
