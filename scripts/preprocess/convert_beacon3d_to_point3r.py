#!/usr/bin/env python3
"""
Convert Beacon3D dataset to Point3R format.
Produces both training and evaluation JSON files.
"""

import json
from pathlib import Path
from tqdm import tqdm


def convert_sample(sample):
    """Convert a single Beacon3D sample to Point3R format."""
    scene_id = sample.get("scene_id")
    if not scene_id:
        return None

    answers = sample.get("answers", [])
    if not answers:
        return None

    # pre-prompt for 3D grounding dataset
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    post_prompt = "Answer the question using a single word or phrase."
    pointer_token = "<|pointer_pad|>"

    question = sample["question"]

    converted = {
        "conversations": [
            {"from": "human", "value": f"{pre_prompt}{pointer_token} {question} {post_prompt}"},
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


def convert_file(input_path, output_path):
    """Convert a Beacon3D JSON file to Point3R format."""
    print(f"Loading: {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    converted = []
    skipped = 0

    for sample in tqdm(data, desc="Converting"):
        result = convert_sample(sample)
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
    base = Path("data")

    # Training set
    convert_file(
        base / "media" / "Beacon3D" / "train.json",
        base / "train" / "beacon3d_train_point3r.json",
    )

    # Evaluation set (val, since test has no answers)
    convert_file(
        base / "media" / "Beacon3D" / "val.json",
        base / "evaluation" / "beacon3d_point3r" / "val.json",
    )


if __name__ == "__main__":
    main()
