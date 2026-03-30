#!/usr/bin/env python3
"""
Convert multi3drefer intermediate JSON to Point3R pointer-based format.

Stage 2 of 2:
  Image-based format → pointer format with gt_bboxes_cam (list of 9-DOF bboxes in camera space)

Input:  data/demo_data/multi3drefer_{split}_32frames_from_frame_0.json
Output: data/train/multi3drefer_train_32frames_point3r.json
        data/evaluation/multi3drefer_point3r/val.json
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def extract_scene_id(image_path):
    """Extract scene_id from image path.
    Example: 'scannet/posed_images/scene0000_00/01140.jpg' -> 'scene0000_00'
    """
    parts = Path(image_path).parts
    if len(parts) >= 3:
        return parts[2]
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
    """Convert a single annotation from image-based to pointer-based format."""
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    pointer_token = "<|pointer_pad|>"
    new_annotation = annotation.copy()

    images = annotation.get('images', [])
    if len(images) == 0:
        print(f"Warning: No images found in annotation {annotation.get('metadata', {})}")
        return None

    scene_id = extract_scene_id(images[0])
    if scene_id is None:
        print(f"Warning: Could not extract scene_id from {images[0]}")
        return None

    new_annotation['pointer_data'] = f"scannet/pointer_memory/{scene_id}.pt"

    # Transform all gt_bboxes to camera space
    gt_bboxes = annotation.get('gt_bboxes')
    cam2global = annotation.get('cam2global')
    axis_align_matrix = annotation.get('axis_align_matrix')
    if gt_bboxes is not None and cam2global is not None and axis_align_matrix is not None:
        extrinsic = np.array(axis_align_matrix) @ np.array(cam2global[0])
        gt_bboxes_cam = [
            [round(x, 4) for x in _9dof_transform_world2cam(b, extrinsic, convention="ZXY")]
            for b in gt_bboxes
        ]
        new_annotation['gt_bboxes_cam'] = gt_bboxes_cam

    # Remove image-based fields
    for key in ('images', 'cam2img', 'cam2global', 'axis_align_matrix'):
        new_annotation.pop(key, None)

    # Replace <image> tokens with pointer token
    if 'conversations' in new_annotation:
        new_conversations = []
        for conv in new_annotation['conversations']:
            new_conv = conv.copy()
            value = conv.get('value', '')
            if '<image>' in value:
                value_without_images = value.replace('<image>', '')
                new_conv['value'] = pre_prompt + pointer_token + " " + value_without_images
            new_conversations.append(new_conv)
        new_annotation['conversations'] = new_conversations

    return new_annotation


def write_dataset_card(output_dir, splits):
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
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return []

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        annotations = json.load(f)

    print(f"Total samples: {len(annotations)}")
    converted, skipped = [], 0
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

    convert_file(
        base / "demo_data" / "multi3drefer_train_32frames_from_frame_0.json",
        base / "train" / "multi3drefer_train_32frames_point3r.json",
    )

    eval_dir = base / "evaluation" / "multi3drefer_point3r"
    convert_file(
        base / "demo_data" / "multi3drefer_val_32frames_from_frame_0.json",
        eval_dir / "val.json",
    )
    write_dataset_card(eval_dir, [("val", "val.json")])


if __name__ == "__main__":
    main()
