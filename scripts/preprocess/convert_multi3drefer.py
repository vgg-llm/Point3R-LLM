#!/usr/bin/env python3
"""
Convert Multi3DRefer dataset to intermediate image-based format for Point3R pipeline.

Stage 1 of 2:
  Raw Multi3DRefer JSON + EmbodiedScan metadata → image-based format with gt_bboxes list

Stage 2: convert_multi3drefer_to_point3r.py
"""

import os
import ray
import json
import torch
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from utils import _9dof_transform_world2cam, uniform_sample_images_custom


def load_scene(filename):
    """Load AABB bounding boxes from ScanNet .pth file.
    Returns dict mapping object_id (str) -> (cx, cy, cz, w, h, l).
    """
    d = torch.load(filename, weights_only=False)
    object_ids = d['aabb_obj_ids'].tolist()
    corner_xyz = d['aabb_corner_xyz'].tolist()

    ret = {}
    for i in range(len(object_ids)):
        object_id = str(object_ids[i])

        xs, ys, zs = zip(*corner_xyz[i])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        l = z_max - z_min

        ret[object_id] = (x_center, y_center, z_center, w, h, l)

    return ret


@ray.remote
def process_split(data, args, split):
    id2scan = {}
    for s in ["train", "val", "test"]:
        pkl_path = os.path.join(args.embodiedscan, f"embodiedscan_infos_{s}.pkl")
        with open(pkl_path, 'rb') as f:
            embodiedscan_data = pickle.load(f)
        id2scan.update({x["sample_idx"]: x for x in embodiedscan_data["data_list"]})

    all_data = []
    scan2box = {}
    missing_num = 0

    for i, item in enumerate(tqdm(data, desc=f"{split}")):
        scene_id = item['scene_id']
        object_ids = item['object_ids']

        # Load GT bboxes (AABB in world coords)
        if split == "test":
            gt_bboxes = None
        else:
            if scene_id not in scan2box:
                pth_path = os.path.join(args.scannet_dir, "pcd_with_object_aabbs", split, f"{scene_id}.pth")
                scan2box[scene_id] = load_scene(pth_path)
            scene_boxes = scan2box[scene_id]
            gt_bboxes = []
            for oid in object_ids:
                oid_str = str(oid)
                if oid_str in scene_boxes:
                    box = list(scene_boxes[oid_str]) + [0, 0, 0]  # 9-DOF AABB
                    gt_bboxes.append(box)
                else:
                    missing_num += 1
                    print(f"Warning: object_id {oid} not found in scene {scene_id}")

        # Load EmbodiedScan scan info
        scan_key = f"scannet/{scene_id}"
        if scan_key not in id2scan:
            print(f"Warning: scan {scan_key} not found in EmbodiedScan")
            continue
        scan = id2scan[scan_key]

        desc = item['description'].capitalize()
        images = uniform_sample_images_custom(scan['images'], args.nframes)

        image_tokens = "".join(["<image>" for _ in range(len(images))])
        prompt = (
            f"Localize all objects matching the following description.\n{desc}\n"
            "Output a JSON dictionary with the 3D bounding boxes in \"bbox_3d\"."
        )
        question = f"{image_tokens}\n{prompt}"

        # Build answer (train only: transform bboxes to camera space)
        if split == "train" and gt_bboxes is not None:
            axis_align_matrix = np.array(scan['axis_align_matrix'])
            reference_frame = images[-1] if args.reference_frame == "last" else images[0]
            extrinsic = axis_align_matrix @ np.array(reference_frame["cam2global"])
            bboxes_cam = [
                [round(x, 2) for x in _9dof_transform_world2cam(b, extrinsic, convention="ZXY")]
                for b in gt_bboxes
            ]
            object_json = {"bbox_3d": bboxes_cam}
            answer = f"```json\n\t{json.dumps(object_json)}\n```"
        else:
            answer = ""

        output = {
            "images": [img["img_path"] for img in images],
            "conversations": [
                {"value": question, "from": "human"},
                {"value": answer, "from": "gpt"},
            ],
            "metadata": {
                "dataset": "multi3drefer",
                "question_type": item["eval_type"],
                "ann_id": item["ann_id"],
                "object_ids": object_ids,
            },
            "gt_bboxes": gt_bboxes,
            "prompt": prompt,
        }

        if split == "val" or args.include_cam_params:
            output.update({
                "cam2img": scan['cam2img'].tolist(),
                "cam2global": [x["cam2global"].tolist() for x in images],
                "axis_align_matrix": scan["axis_align_matrix"].tolist(),
            })

        all_data.append(output)

    return all_data, missing_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi3drefer_dir", type=str, default="data/benchmark/multi3drefer/")
    parser.add_argument("--scannet_dir", type=str, default="./data/media/scannet")
    parser.add_argument("--embodiedscan", type=str, default="/home/gwakcy/datasets/embodiedscan-v2")
    parser.add_argument("--reference_frame", type=str, default="first")
    parser.add_argument("--output_dir", type=str, default="data/demo_data")
    parser.add_argument("--nframes", type=int, default=32)
    parser.add_argument("--include_cam_params", action="store_true")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    for split in ["train", "val"]:
        json_path = os.path.join(args.multi3drefer_dir, f"multi3drefer_{split}.json")
        with open(json_path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items for split={split}")

        futures = []
        chunk_size = max(1, len(data) // 8)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        for chunk in chunks:
            futures.append(process_split.remote(chunk, args, split))

        all_data = []
        total_missing = 0
        for result in ray.get(futures):
            chunk_data, missing = result
            all_data.extend(chunk_data)
            total_missing += missing

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"multi3drefer_{split}_{args.nframes}frames_from_frame_0.json")
        with open(output_path, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"Saved {len(all_data)} items to {output_path} (missing: {total_missing})")

    ray.shutdown()


if __name__ == "__main__":
    main()
