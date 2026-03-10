#!/usr/bin/env python3
"""
Convert scanrefer_train_32frames.json to scanrefer_train_32frames_point3r.json
Replaces multiple <image> tokens with pointer tokens and pointer_data paths.
Removes camera parameter fields (cam2img, cam2global, axis_align_matrix).
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def extract_scene_id(image_path):
    """
    Extract scene_id from image path.
    Example: 'scannet/posed_images/scene0000_00/01140.jpg' -> 'scene0000_00'
    """
    parts = Path(image_path).parts
    if len(parts) >= 3:
        return parts[2]  # scene0000_00
    return None


def _9dof_transform_world2cam(box, extrinsic, convention):
    """Transform a 9-DOF bbox from world coords to camera coords."""
    center = box[:3]
    extent = box[3:6]
    euler = box[6:]

    global2cam = np.linalg.inv(extrinsic)
    new_center = (global2cam @ np.array(list(center) + [1]).reshape(4, 1)).reshape(4)[:3].tolist()
    new_rot = global2cam[:3, :3] @ R.from_euler(convention, euler).as_matrix()
    new_euler = R.from_matrix(new_rot).as_euler(convention).tolist()
    return new_center + list(extent) + new_euler


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

    # Compute gt_bbox_cam: transform gt_bbox from world to reference frame (first frame) camera coords
    gt_bbox = annotation.get('gt_bbox')
    cam2global = annotation.get('cam2global')
    axis_align_matrix = annotation.get('axis_align_matrix')
    if gt_bbox is not None and cam2global is not None and axis_align_matrix is not None:
        extrinsic = np.array(axis_align_matrix) @ np.array(cam2global[0])  # reference frame = first
        gt_bbox_cam = _9dof_transform_world2cam(gt_bbox, extrinsic, convention="ZXY")
        new_annotation['gt_bbox_cam'] = [round(x, 4) for x in gt_bbox_cam]

    # Remove image-based fields
    for key in ('images', 'cam2img', 'cam2global', 'axis_align_matrix'):
        if key in new_annotation:
            del new_annotation[key]

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
    """Convert a scanrefer JSON file to Point3R format."""
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
        base / "demo_data" / "scanrefer_train_32frames_from_frame_0.json",
        base / "train" / "scanrefer_train_32frames_point3r.json",
    )

    # Evaluation set (val)
    eval_dir = base / "evaluation" / "scanrefer_point3r"
    convert_file(
        base / "demo_data" / "scanrefer_val_32frames_from_frame_0.json",
        eval_dir / "val.json",
    )
    write_dataset_card(eval_dir, [("val", "val.json")])


if __name__ == "__main__":
    main()
