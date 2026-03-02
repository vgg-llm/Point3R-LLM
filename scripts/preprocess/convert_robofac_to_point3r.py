#!/usr/bin/env python3
"""
Convert RoboFAC dataset to Point3R format.

Training data:
  Replaces <video> tokens with <|pointer_pad|> and adds pointer_data paths.
  Input:  data/media/robofac/training_qa.json
  Output: data/train/robofac_training_qa_point3r.json

Eval data:
  Adds pointer_data field to flattened test entries.
  Input:  data/robofac/test.json
  Output: data/evaluation/robofac_point3r/test.json

Usage:
    python scripts/preprocess/convert_robofac_to_point3r.py
    python scripts/preprocess/convert_robofac_to_point3r.py --pointer-dir pointer_memory_qwen3vl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_training_annotation(annotation, pointer_dir):
    """
    Convert a single training annotation from video-based to pointer-based format.

    Original format:
        {"id": "...", "video": "TaskName/.../uuid.mp4", "conversations": [...]}
    Converted format:
        {"id": "...", "pointer_data": "robofac/{pointer_dir}/simulation_data/TaskName/.../uuid.pt",
         "conversations": [...]}
    """
    pointer_token = "<|pointer_pad|>"
    new_annotation = annotation.copy()

    video_path = annotation["video"]

    # Build pointer_data path: robofac/{pointer_dir}/simulation_data/{video_relative}.pt
    pointer_data_path = f"robofac/{pointer_dir}/simulation_data/{Path(video_path).with_suffix('.pt')}"
    new_annotation["pointer_data"] = pointer_data_path

    # Remove video field
    del new_annotation["video"]

    # Replace <video> with <|pointer_pad|> in conversations
    new_conversations = []
    for conv in new_annotation["conversations"]:
        new_conv = conv.copy()
        value = conv["value"]

        if "<video>" in value:
            # Replace <video> token with pointer token
            value = value.replace("<video>", pointer_token)
            new_conv["value"] = value

        new_conversations.append(new_conv)

    new_annotation["conversations"] = new_conversations
    return new_annotation


def convert_eval_entry(entry, pointer_dir):
    """
    Add pointer_data field to a flattened eval entry.

    Original: {"video_path": "simulation_data/TaskName/.../uuid.mp4", ...}
    Added:    {"pointer_data": "robofac/{pointer_dir}/simulation_data/TaskName/.../uuid.pt", ...}
    """
    new_entry = entry.copy()
    video_path = entry["video_path"]

    # Build pointer_data path: robofac/{pointer_dir}/{video_path}.pt
    pointer_data_path = f"robofac/{pointer_dir}/{Path(video_path).with_suffix('.pt')}"
    new_entry["pointer_data"] = pointer_data_path

    return new_entry


def convert_training_data(input_path, output_path, pointer_dir):
    """Convert training annotations."""
    print(f"Loading training annotations from: {input_path}")
    with open(input_path, "r") as f:
        annotations = json.load(f)
    print(f"Total input annotations: {len(annotations)}")

    converted = []
    for annotation in tqdm(annotations, desc="Converting training data"):
        converted.append(convert_training_annotation(annotation, pointer_dir))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted: {len(converted)}")

    # Print sample
    if converted:
        print("\n" + "=" * 80)
        print("SAMPLE CONVERSION (First annotation)")
        print("=" * 80)
        orig = annotations[0]
        conv = converted[0]
        print(f"\n--- Original ---")
        print(f"Video: {orig['video']}")
        print(f"Human: {orig['conversations'][0]['value'][:150]}...")
        print(f"\n--- Converted ---")
        print(f"Pointer data: {conv['pointer_data']}")
        print(f"Human: {conv['conversations'][0]['value'][:150]}...")
        print("=" * 80)


def convert_eval_data(input_path, output_path, pointer_dir):
    """Convert eval annotations."""
    print(f"\nLoading eval data from: {input_path}")
    with open(input_path, "r") as f:
        entries = json.load(f)
    print(f"Total eval entries: {len(entries)}")

    converted = []
    for entry in tqdm(entries, desc="Converting eval data"):
        converted.append(convert_eval_entry(entry, pointer_dir))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted: {len(converted)}")

    # Print sample
    if converted:
        print("\n" + "=" * 80)
        print("SAMPLE EVAL CONVERSION (First entry)")
        print("=" * 80)
        orig = entries[0]
        conv = converted[0]
        print(f"\n--- Original ---")
        print(f"Video path: {orig['video_path']}")
        print(f"\n--- Converted ---")
        print(f"Video path: {conv['video_path']}")
        print(f"Pointer data: {conv['pointer_data']}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboFAC dataset to Point3R format"
    )
    parser.add_argument(
        "--pointer-dir",
        type=str,
        default="pointer_memory",
        help="Pointer memory directory name (default: pointer_memory)",
    )
    parser.add_argument(
        "--train-input",
        type=str,
        default="data/media/robofac/training_qa.json",
        help="Input training annotation file",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/train/robofac_training_qa_point3r.json",
        help="Output training annotation file",
    )
    parser.add_argument(
        "--eval-input",
        type=str,
        default="data/robofac/test.json",
        help="Input eval data file",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="data/evaluation/robofac_point3r/test.json",
        help="Output eval data file",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training data conversion",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval data conversion",
    )
    args = parser.parse_args()

    if not args.skip_train:
        if Path(args.train_input).exists():
            convert_training_data(args.train_input, args.train_output, args.pointer_dir)
        else:
            print(f"Warning: Training input not found: {args.train_input}")

    if not args.skip_eval:
        if Path(args.eval_input).exists():
            convert_eval_data(args.eval_input, args.eval_output, args.pointer_dir)
        else:
            print(f"Warning: Eval input not found: {args.eval_input}")

    print("\nDone!")


if __name__ == "__main__":
    main()
