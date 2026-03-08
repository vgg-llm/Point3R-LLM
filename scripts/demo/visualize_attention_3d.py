"""
Visualize LLM attention on 3D point clouds using viser.

Loads pre-computed pointer_data from a .pt file, runs LLM inference with
attention capture, then launches an interactive viser visualization where
dense point clouds are colored by the attention each region receives.

Usage:
    python scripts/demo/visualize_attention_3d.py \
        --pointer_data_path ./data/demo_data/scene0706_00_32f_video_compact.pt \
        --query "How many are the pillows?" \
        --model_path Qwen/Qwen3-VL-4B-Instruct \
        [--layer_indices -8 -7 -6 -5 -4 -3 -2 -1] \
        [--max_new_tokens 128]
"""

import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/demo")

from demo_point3r import load_models, set_seed
from visualize_attention import run_models_with_attention
from qwen_vl.model.point3r.extract_memory import visualize_point3r_viser
from qwen_vl.model.point3r.inference import get_pred_pts3d


def main():
    parser = argparse.ArgumentParser(description="3D attention visualization on Point3R point clouds")
    parser.add_argument("--pointer_data_path", type=str, required=True,
                        help="Path to pre-computed pointer data .pt file")
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
    args = parser.parse_args()

    set_seed(args.seed)

    # Load pointer data
    print(f"Loading pointer data from {args.pointer_data_path}...")
    pointer_data = torch.load(args.pointer_data_path, weights_only=False)
    print("Pointer data loaded successfully!")

    if '_point3r_outputs' not in pointer_data:
        print("ERROR: pointer_data does not contain '_point3r_outputs'.")
        print("The .pt file must include full Point3R outputs for dense point cloud visualization.")
        sys.exit(1)

    # Load model with eager attention for attention weight extraction
    print("\nLoading model with attn_implementation='eager'...")
    model, processor, min_pixels, max_pixels, _ = load_models(
        load_point3r=False,
        model_path=args.model_path,
        pointer_format=args.pointer_format,
        use_merge=args.use_merge,
        attn_implementation="eager",
    )

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

    attn_result["attention_matrix"][:, 50:150] = 0

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
