#!/usr/bin/env python3
"""
Convert SPAR-subset training and evaluation data to Point3R format.
Handles both ScanNet and ScanNet++ data in "fill" and "sentence" variants.

Source data: data/SPAR-subset/train/*.jsonl and data/SPAR-subset/val/*.jsonl
Training output: data/train/spar_subset_point3r.json
Evaluation output: data/evaluation/spar_subset_point3r/test.json

Pointer memory paths:
- ScanNet: scannet/pointer_memory/{scene_name}.pt
- ScanNet++: scannetpp/pointer_memory/{scene_name}.pt
"""

import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm


POINTER_MEMORY_PATHS = {
    "scannet": "scannet/pointer_memory",
    "scannetpp": "scannetpp/pointer_memory",
}


def detect_data_source(filename):
    """Detect data source (scannet/scannetpp) from filename."""
    if filename.startswith("scannetpp_"):
        return "scannetpp"
    elif filename.startswith("scannet_"):
        return "scannet"
    return None


def detect_variant(filename):
    """Detect variant (fill/sentence) from filename."""
    if "_fill." in filename or "_fill_" in filename:
        return "fill"
    elif "_sentence." in filename or "_sentence_" in filename:
        return "sentence"
    return None


def extract_scene_name(sample):
    """Extract scene name from image paths."""
    images = sample.get("image", [])
    if images:
        return images[0].split("/")[0]
    # Fallback: use id with rsplit
    sample_id = sample.get("id", "")
    if sample_id:
        return sample_id.rsplit("_", 1)[0]
    return None


def convert_train_sample(sample, data_source, variant, num_pointer_tokens=1):
    """Convert a single SPAR-subset sample to Point3R training format."""
    scene_name = extract_scene_name(sample)
    if not scene_name:
        return None

    if data_source not in POINTER_MEMORY_PATHS:
        return None

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    # Build pointer sequence
    pointer_sequence = (
        "<|vision_start|>" +
        "<|pointer_pad|>" * num_pointer_tokens +
        "<|vision_end|>"
    )

    # Process conversations
    new_conversations = []
    for conv in sample.get("conversations", []):
        new_conv = conv.copy()
        value = conv.get("value", "")

        if conv.get("from") == "human":
            value = value.replace("<image>", "").strip()
            new_conv["value"] = f"{pointer_sequence}\n{value}"

        new_conversations.append(new_conv)

    converted = {
        "conversations": new_conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "metadata": {
            "dataset": "spar_subset",
            "data_source": data_source,
            "variant": variant,
            "scene_id": scene_name,
            "original_id": sample.get("id", ""),
        },
    }
    return converted


def convert_eval_sample(sample, data_source, variant, num_pointer_tokens=1):
    """Convert a single SPAR-subset sample to Point3R evaluation format."""
    scene_name = extract_scene_name(sample)
    if not scene_name:
        return None

    if data_source not in POINTER_MEMORY_PATHS:
        return None

    pointer_memory_prefix = POINTER_MEMORY_PATHS[data_source]

    # Build pointer sequence
    pointer_sequence = (
        "<|vision_start|>" +
        "<|pointer_pad|>" * num_pointer_tokens +
        "<|vision_end|>"
    )

    # Process conversations
    new_conversations = []
    for conv in sample.get("conversations", []):
        new_conv = conv.copy()
        value = conv.get("value", "")

        if conv.get("from") == "human":
            value = value.replace("<image>", "").strip()
            new_conv["value"] = f"{pointer_sequence}\n{value}"

        new_conversations.append(new_conv)

    # Ground truth is the GPT response
    ground_truth = ""
    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value", "")
            break

    converted = {
        "id": sample.get("id"),
        "conversations": new_conversations,
        "pointer_data": f"{pointer_memory_prefix}/{scene_name}.pt",
        "ground_truth": ground_truth,
        "question_type": variant,
        "data_source": data_source,
        "metadata": {
            "dataset": "spar_subset",
            "scene_id": scene_name,
        },
    }
    return converted


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_split(source_dir, split_name, convert_fn, num_pointer_tokens,
                  enabled_sources):
    """Process all JSONL files in a split directory."""
    all_converted = []
    stats = {
        "scannet_fill": 0,
        "scannet_sentence": 0,
        "scannetpp_fill": 0,
        "scannetpp_sentence": 0,
        "skipped": 0,
    }

    jsonl_files = sorted([
        f for f in os.listdir(source_dir) if f.endswith(".jsonl")
    ])

    for jsonl_file in jsonl_files:
        data_source = detect_data_source(jsonl_file)
        variant = detect_variant(jsonl_file)

        if data_source is None or variant is None:
            print(f"Warning: Could not detect source/variant from {jsonl_file}, skipping")
            continue

        if enabled_sources and data_source not in enabled_sources:
            print(f"Skipping {jsonl_file} (source {data_source} not enabled)")
            continue

        input_path = source_dir / jsonl_file
        print(f"Loading: {input_path}")
        data = load_jsonl(input_path)
        print(f"  Entries: {len(data)}")

        for sample in tqdm(data, desc=f"Converting {jsonl_file}"):
            result = convert_fn(
                sample, data_source, variant, num_pointer_tokens
            )
            if result is not None:
                all_converted.append(result)
                stats_key = f"{data_source}_{variant}"
                if stats_key in stats:
                    stats[stats_key] += 1
            else:
                stats["skipped"] += 1

    return all_converted, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert SPAR-subset to Point3R format (train + eval)"
    )
    parser.add_argument(
        "--num_pointer_tokens",
        type=int,
        default=1,
        help="Number of pointer tokens per sample",
    )
    parser.add_argument(
        "--exclude_scannetpp",
        action="store_true",
        help="Exclude ScanNet++ data (included by default)",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default=None,
        help="Training output path (default: data/train/spar_subset_point3r.json)",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Eval output path (default: data/evaluation/spar_subset_point3r/test.json)",
    )
    args = parser.parse_args()

    base = Path("data")
    train_dir = base / "SPAR-subset" / "train"
    val_dir = base / "SPAR-subset" / "val"

    # Determine which data sources to include
    enabled_sources = {"scannet", "scannetpp"}
    if args.exclude_scannetpp:
        enabled_sources.discard("scannetpp")

    print(f"Enabled data sources: {enabled_sources}")

    # --- Process training data ---
    print("\n" + "=" * 60)
    print("PROCESSING TRAINING DATA")
    print("=" * 60)

    train_converted, train_stats = process_split(
        train_dir, "train", convert_train_sample,
        args.num_pointer_tokens, enabled_sources,
    )

    train_output = Path(args.train_output) if args.train_output else (
        base / "train" / "spar_subset_point3r.json"
    )
    train_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving training data to: {train_output}")
    with open(train_output, "w") as f:
        json.dump(train_converted, f, indent=2)

    print(f"\nTraining Conversion Statistics:")
    for key, count in sorted(train_stats.items()):
        print(f"  {key}: {count}")
    print(f"  Total converted: {len(train_converted)}")

    # --- Process evaluation data ---
    print("\n" + "=" * 60)
    print("PROCESSING EVALUATION DATA")
    print("=" * 60)

    eval_converted, eval_stats = process_split(
        val_dir, "val", convert_eval_sample,
        args.num_pointer_tokens, enabled_sources,
    )

    eval_output = Path(args.eval_output) if args.eval_output else (
        base / "evaluation" / "spar_subset_point3r" / "test.json"
    )
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving evaluation data to: {eval_output}")
    with open(eval_output, "w") as f:
        json.dump(eval_converted, f, indent=2)

    print(f"\nEvaluation Conversion Statistics:")
    for key, count in sorted(eval_stats.items()):
        print(f"  {key}: {count}")
    print(f"  Total converted: {len(eval_converted)}")

    # --- Print samples ---
    for label, data in [("TRAINING", train_converted), ("EVALUATION", eval_converted)]:
        if len(data) > 0:
            print(f"\n{'=' * 60}")
            print(f"SAMPLE {label} CONVERSIONS")
            print("=" * 60)
            shown = set()
            for item in data:
                variant = item.get("question_type") or item.get("metadata", {}).get("variant", "")
                ds = item.get("data_source") or item.get("metadata", {}).get("data_source", "")
                key = f"{ds}_{variant}"
                if key not in shown:
                    shown.add(key)
                    print(f"\n--- {key.upper()} ---")
                    print(json.dumps(item, indent=2)[:800])
                    if len(shown) >= 4:
                        break


if __name__ == "__main__":
    main()
