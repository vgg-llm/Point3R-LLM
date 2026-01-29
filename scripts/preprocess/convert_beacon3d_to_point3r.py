#!/usr/bin/env python3
"""
Convert Beacon3D dataset to Point3R format.
Produces both training and evaluation JSON files.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_sample(sample, num_pointer_tokens=1):
    """Convert a single Beacon3D sample to Point3R format."""
    scene_id = sample.get("scene_id")
    if not scene_id:
        return None

    answers = sample.get("answers", [])
    if not answers:
        return None

    pointer_sequence = (
        "<|vision_start|>" +
        "<|pointer_pad|>" * num_pointer_tokens +
        "<|vision_end|>"
    )

    question = sample["question"]

    converted = {
        "conversations": [
            {"from": "human", "value": f"{pointer_sequence}\n{question}\nAnswer in a single word or phrase."},
            {"from": "gpt", "value": answers[0]},
        ],
        "pointer_data": f"scannet/pointer_memory/{scene_id}.pt",
        "answers": answers,
        "metadata": {
            "dataset": "beacon3d",
            "question_id": sample.get("question_id", ""),
            "scene_id": scene_id,
        },
    }
    return converted


def convert_file(input_path, output_path, num_pointer_tokens=1):
    """Convert a Beacon3D JSON file to Point3R format."""
    print(f"Loading: {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    converted = []
    skipped = 0

    for sample in tqdm(data, desc="Converting"):
        result = convert_sample(sample, num_pointer_tokens)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving: {output_path}")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"  Converted: {len(converted)}, Skipped: {skipped}")
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert Beacon3D to Point3R format"
    )
    parser.add_argument(
        "--num_pointer_tokens",
        type=int,
        default=1,
        help="Number of pointer tokens per sample",
    )
    args = parser.parse_args()

    base = Path("data")

    # Training set
    convert_file(
        base / "media" / "Beacon3D" / "train.json",
        base / "train" / "beacon3d_train_point3r.json",
        args.num_pointer_tokens,
    )

    # Evaluation set (val, since test has no answers)
    convert_file(
        base / "media" / "Beacon3D" / "val.json",
        base / "evaluation" / "beacon3d_point3r" / "val.json",
        args.num_pointer_tokens,
    )


if __name__ == "__main__":
    main()
