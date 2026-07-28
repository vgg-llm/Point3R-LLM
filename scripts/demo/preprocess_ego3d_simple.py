#!/usr/bin/env python
"""Extract Point3R pointer memory for Ego3D-Bench scenes.

Run one process per GPU:
    CUDA_VISIBLE_DEVICES=0 python preprocess_ego3d_simple.py --gpu-id 0 --total-gpus 4
    CUDA_VISIBLE_DEVICES=1 python preprocess_ego3d_simple.py --gpu-id 1 --total-gpus 4

262 scenes at 5-7 views each; a single GPU finishes in minutes.
"""

import argparse
import json
from pathlib import Path

from demo_point3r import load_models, preprocess_images
from tqdm import tqdm


def setup_ego3d_paths(scenes_json, scenes_root, save_path):
    scenes = json.loads(Path(scenes_json).read_text())
    scene_dirs = [str(Path(scenes_root) / s["scene_id"]) for s in scenes]
    out_paths = [str(Path(save_path) / f"{s['scene_id']}.pt") for s in scenes]
    return scene_dirs, out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU index (0-based)")
    parser.add_argument("--total-gpus", type=int, required=True, help="number of GPUs")
    parser.add_argument("--scenes-json", default="data/evaluation/ego3d_point3r/scenes.json")
    parser.add_argument("--scenes-root", default="data/media/ego3d/scenes")
    parser.add_argument("--save-path", default="data/media/ego3d/pointer_memory_qwen3vl")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lambda-decay", type=float, default=1.0)
    parser.add_argument("--sample-ct", type=int, default=32)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--len-unit", type=int, default=20)
    parser.add_argument("--no-merge", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    scene_dirs, out_paths = setup_ego3d_paths(
        args.scenes_json, args.scenes_root, args.save_path
    )
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    local_dirs = scene_dirs[args.gpu_id::args.total_gpus]
    local_outs = out_paths[args.gpu_id::args.total_gpus]
    print(f"GPU {args.gpu_id}/{args.total_gpus - 1}: {len(local_dirs)} of "
          f"{len(scene_dirs)} scenes")

    model, processor, min_pixels, max_pixels, point3r_model = load_models(
        device=None,
        model_path=args.model_path,
        use_merge=not args.no_merge,
        merge_threshold=args.merge_threshold,
        len_unit=args.len_unit,
    )

    failures = []
    for scene_dir, out_path in tqdm(
        zip(local_dirs, local_outs), desc=f"GPU {args.gpu_id}", total=len(local_dirs)
    ):
        if Path(out_path).exists() and not args.overwrite:
            continue
        # Write to a temp path, then rename, so an interrupted run cannot leave a
        # truncated .pt that would later load as valid-looking garbage.
        tmp_path = f"{out_path}.tmp"
        try:
            preprocess_images(
                model, processor, min_pixels, max_pixels, point3r_model,
                scene_dir, tmp_path,
                use_viser=False, unload_point3r_model=False,
                lambda_decay=args.lambda_decay, sample_ct=args.sample_ct,
                image_extensions=("*.jpg",),
            )
            Path(tmp_path).rename(out_path)
        except Exception as exc:  # noqa: BLE001 - one bad scene must not kill the shard
            print(f"\nFAILED {scene_dir}: {exc}")
            Path(tmp_path).unlink(missing_ok=True)
            failures.append(scene_dir)

    print(f"GPU {args.gpu_id}: done, {len(failures)} failures")
    for scene_dir in failures:
        print(f"  failed: {scene_dir}")


if __name__ == "__main__":
    main()
