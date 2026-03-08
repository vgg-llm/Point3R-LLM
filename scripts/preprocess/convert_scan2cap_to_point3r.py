#!/usr/bin/env python3
"""
Convert scan2cap_train_32frames.json to scan2cap_train_32frames_point3r.json
Replaces multiple <image> tokens with pointer tokens and pointer_data paths.
"""

import json
from pathlib import Path
from tqdm import tqdm


def extract_scene_id(image_path):
    """
    Extract scene_id from image path.
    Example: 'scannet/posed_images/scene0000_00/01140.jpg' -> 'scene0000_00'
    """
    parts = Path(image_path).parts
    if len(parts) >= 3:
        return parts[2]  # scene0000_00
    return None


def convert_annotation(annotation):
    """
    Convert a single annotation from image-based to pointer-based format.

    Args:
        annotation: Original annotation dict with 'images' and <image> tokens

    Returns:
        Converted annotation dict with 'pointer_data' and pointer tokens
    """
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    pointer_token = "<|pointer_pad|>"
    new_annotation = annotation.copy()

    # Extract scene_id from first image path
    images = annotation.get('images', [])
    if len(images) == 0:
        print(f"Warning: No images found in annotation {annotation.get('metadata', {})}")
        return None

    scene_id = extract_scene_id(images[0])
    if scene_id is None:
        print(f"Warning: Could not extract scene_id from {images[0]}")
        return None

    # Create pointer_data path
    pointer_data_path = f"scannet/pointer_memory/{scene_id}.pt"
    new_annotation['pointer_data'] = pointer_data_path

    # Remove 'images' field
    if 'images' in new_annotation:
        del new_annotation['images']

    # Update conversations to replace <image> tokens with pointer token
    if 'conversations' in new_annotation:
        new_conversations = []
        for conv in new_annotation['conversations']:
            new_conv = conv.copy()
            value = conv.get('value', '')

            # Replace <image> tokens with a single pointer token
            num_images = value.count('<image>')
            if num_images > 0:
                value_without_images = value.replace('<image>', '')
                new_value = pre_prompt + pointer_token + " " + value_without_images
                new_conv['value'] = new_value

            new_conversations.append(new_conv)

        new_annotation['conversations'] = new_conversations

    return new_annotation


def write_dataset_card(output_dir, splits):
    """Write a HuggingFace dataset card (README.md) with explicit split definitions."""
    data_files = "\n".join(
        f"  - split: {split}\n    path: {filename}" for split, filename in splits
    )
    readme = f"""---
license: apache-2.0
configs:
- config_name: default
  data_files:
{data_files}
---
"""
    with open(Path(output_dir) / "README.md", "w") as f:
        f.write(readme)


def convert_file(input_path, output_path):
    """Convert a scan2cap JSON file to Point3R format."""
    print(f"Loading: {input_path}")
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return []

    with open(input_path, 'r') as f:
        annotations = json.load(f)

    print(f"Total samples: {len(annotations)}")
    converted = []
    skipped = 0

    for annotation in tqdm(annotations, desc="Converting"):
        result = convert_annotation(annotation)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"  Converted: {len(converted)}, Skipped: {skipped}")
    return converted


def main():
    base = Path("data")

    # Training set
    convert_file(
        base / "train" / "scan2cap_train_32frames.json",
        base / "train" / "scan2cap_train_32frames_point3r.json",
    )

    # Evaluation set (val)
    eval_dir = base / "evaluation" / "scan2cap_point3r"
    convert_file(
        base / "evaluation" / "scan2cap" / "scan2cap_val_32frames.json",
        eval_dir / "val.json",
    )
    write_dataset_card(eval_dir, [("val", "val.json")])


if __name__ == "__main__":
    main()
