"""
Visualize LLM attention on 3D point clouds using viser.

Supports three input modes:
  1. Pre-computed pointer_data from a .pt file (--pointer_data_path)
  2. Raw images from a directory (--input_images_dir)
  3. Raw video file (--input_video)

For modes 2 and 3, Point3R preprocessing runs automatically and results are
cached to --output_dir (default: data/demo_data/visualization/). A registry.json
file tracks the mapping from input paths to cached .pt filenames.

Usage:
    # From pre-computed .pt file
    python scripts/demo/visualize_attention_3d.py \
        --pointer_data_path ./data/demo_data/scene0706_00_32f_video_compact.pt \
        --query "How many are the pillows?"

    # From raw images
    python scripts/demo/visualize_attention_3d.py \
        --input_images_dir ./data/demo_data/demo_photos/ \
        --query "Describe this scene."

    # From video
    python scripts/demo/visualize_attention_3d.py \
        --input_video ./data/demo_data/robofac_demo.mp4 \
        --query "What objects are on the table?"
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/demo")

from demo_point3r import load_models, set_seed, preprocess_images, preprocess_video
from visualize_attention import run_models_with_attention
from qwen_vl.model.point3r.extract_memory import visualize_point3r_viser
from qwen_vl.model.point3r.inference import get_pred_pts3d


def resolve_pointer_data_path(input_path, output_dir):
    """Look up or assign a cached .pt filename for the given input path.

    Returns (pointer_data_path, already_exists).
    """
    registry_path = os.path.join(output_dir, "registry.json")
    abs_input = os.path.abspath(input_path)

    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"next_id": 1, "entries": {}}

    if abs_input in registry["entries"]:
        pt_name = registry["entries"][abs_input]
        pt_path = os.path.join(output_dir, pt_name)
        if os.path.exists(pt_path):
            return pt_path, True
        # File was deleted; will re-generate with same name
        return pt_path, False

    # Assign new name
    pt_name = f"vis_{registry['next_id']:04d}.pt"
    registry["entries"][abs_input] = pt_name
    registry["next_id"] += 1

    os.makedirs(output_dir, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return os.path.join(output_dir, pt_name), False


def main():
    parser = argparse.ArgumentParser(description="3D attention visualization on Point3R point clouds")
    parser.add_argument("--pointer_data_path", type=str, default=None,
                        help="Path to pre-computed pointer data .pt file")
    parser.add_argument("--input_images_dir", type=str, default=None,
                        help="Path to directory of images to process with Point3R")
    parser.add_argument("--input_video", type=str, default=None,
                        help="Path to video file to process with Point3R")
    parser.add_argument("--output_dir", type=str, default="data/demo_data/visualization",
                        help="Directory for cached .pt files and registry (default: data/demo_data/visualization)")
    parser.add_argument("--sample_ct", type=int, default=32,
                        help="Number of frames to sample from images/video (default: 32)")
    parser.add_argument("--query", type=str, default="Describe this scene.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--pointer_format", type=str, default="video")
    parser.add_argument("--use_merge", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer_indices", type=int, nargs="+", default=None,
                        help="Layer indices for attention aggregation (e.g. -8 -7 ... -1). Default: all layers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Fast mode: downsampled points, no GUI controls, no attention pre-computation")
    parser.add_argument("--point_stride", type=int, default=1,
                        help="Spatial downsampling stride for point clouds (1=full, 4=16x fewer)")
    parser.add_argument("--scannet", action="store_true", default=False,
                        help="ScanNet mode: only load *.jpg images")
    args = parser.parse_args()

    # Validate input arguments
    has_raw_input = args.input_images_dir is not None or args.input_video is not None
    if args.pointer_data_path is None and not has_raw_input:
        parser.error("Provide either --pointer_data_path, --input_images_dir, or --input_video")
    if args.input_images_dir is not None and args.input_video is not None:
        parser.error("Cannot specify both --input_images_dir and --input_video")

    set_seed(args.seed)

    need_preprocess = args.pointer_data_path is None

    if not need_preprocess:
        # Existing path: load .pt file directly, load LLM only
        print(f"Loading pointer data from {args.pointer_data_path}...")
        pointer_data = torch.load(args.pointer_data_path, weights_only=False)
        print("Pointer data loaded successfully!")

        if '_point3r_outputs' not in pointer_data:
            print("ERROR: pointer_data does not contain '_point3r_outputs'.")
            print("The .pt file must include full Point3R outputs for dense point cloud visualization.")
            sys.exit(1)

        print("\nLoading model with attn_implementation='eager'...")
        model, processor, min_pixels, max_pixels, _ = load_models(
            load_point3r=False,
            model_path=args.model_path,
            pointer_format=args.pointer_format,
            use_merge=args.use_merge,
            attn_implementation="eager",
        )
    else:
        # Resolve input to a cached .pt path via registry
        input_path = args.input_images_dir or args.input_video
        pointer_data_path, cached = resolve_pointer_data_path(input_path, args.output_dir)

        if cached:
            print(f"Reusing cached pointer data: {pointer_data_path}")
            pointer_data = torch.load(pointer_data_path, weights_only=False)

            if '_point3r_outputs' not in pointer_data:
                print("ERROR: cached pointer_data does not contain '_point3r_outputs'.")
                sys.exit(1)

            print("\nLoading model with attn_implementation='eager'...")
            model, processor, min_pixels, max_pixels, _ = load_models(
                load_point3r=False,
                model_path=args.model_path,
                pointer_format=args.pointer_format,
                use_merge=args.use_merge,
                attn_implementation="eager",
            )
        else:
            # Load both Point3R and LLM for preprocessing + attention
            print(f"\nProcessing raw input, output will be saved to: {pointer_data_path}")
            print("Loading Point3R + LLM with attn_implementation='eager'...")
            model, processor, min_pixels, max_pixels, point3r_model = load_models(
                load_point3r=True,
                model_path=args.model_path,
                pointer_format=args.pointer_format,
                use_merge=args.use_merge,
                attn_implementation="eager",
            )

            if args.input_images_dir is not None:
                print(f"\nPreprocessing images from {args.input_images_dir}...")
                extra_kwargs = {}
                if args.scannet:
                    extra_kwargs["image_extensions"] = ("*.jpg",)
                pointer_data = preprocess_images(
                    model, processor, min_pixels, max_pixels, point3r_model,
                    input_images_dir=args.input_images_dir,
                    pointer_data_path=pointer_data_path,
                    save_point3r_outputs=True,
                    sample_ct=args.sample_ct,
                    **extra_kwargs,
                )
            else:
                print(f"\nPreprocessing video from {args.input_video}...")
                pointer_data = preprocess_video(
                    model, processor, min_pixels, max_pixels, point3r_model,
                    input_video_path=args.input_video,
                    pointer_data_path=pointer_data_path,
                    save_point3r_outputs=True,
                    sample_ct=args.sample_ct,
                )

            if '_point3r_outputs' not in pointer_data:
                print("ERROR: preprocessing did not produce '_point3r_outputs'.")
                sys.exit(1)

            print(f"Pointer data saved to {pointer_data_path}")

            # Free Point3R VRAM
            del point3r_model
            torch.cuda.empty_cache()

    # Run inference with attention capture
    print("\nRunning inference with attention capture...")
    attn_result = run_models_with_attention(
        model=model,
        processor=processor,
        pointer_data=pointer_data,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
        layer_indices=args.layer_indices,
    )
    print(f"\nGenerated text: {attn_result['generated_text']}")
    print(f"Attention matrix shape: {attn_result['attention_matrix'].shape}")

    # Build attention_data dict for viser
    ptr_pos = pointer_data['pointer_positions'].numpy()
    ptr_ts = pointer_data['pointer_timestamps'].numpy() if 'pointer_timestamps' in pointer_data else np.zeros(len(ptr_pos))

    attn_result["attention_matrix"][:, 0:200] = 0

    attention_data = {
        "attention_matrix": attn_result["attention_matrix"],
        "generated_tokens_text": attn_result["generated_tokens_text"],
        "pointer_positions": ptr_pos,
        "pointer_timestamps": ptr_ts,
    }

    # Launch viser with attention overlay
    print("\nLaunching viser 3D visualization with attention overlay...")
    visualize_point3r_viser(
        pointer_data,
        attention_data=attention_data,
        fast=args.fast,
        point_stride=args.point_stride,
    )


if __name__ == "__main__":
    main()
