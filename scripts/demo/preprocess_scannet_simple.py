#!/usr/bin/env python
"""
Simple parallel preprocessing script that can be run multiple times with different GPU assignments.
Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess_scannet_simple.py --gpu-id 0 --total-gpus 8
    CUDA_VISIBLE_DEVICES=1 python preprocess_scannet_simple.py --gpu-id 1 --total-gpus 8
    ...
"""

import argparse
from pathlib import Path
from demo_point3r import load_models, preprocess_images
from tqdm import tqdm

def setup_scannet_paths():
    base_dir = Path('./data/media/scannet')
    base_dir.mkdir(parents=True, exist_ok=True)

    posed_images_dir = base_dir / 'posed_images'
    input_image_paths = sorted([str(subfolder) for subfolder in posed_images_dir.iterdir() if subfolder.is_dir()])

    pointer_memory_dir = base_dir / 'pointer_memory'
    pointer_memory_dir.mkdir(parents=True, exist_ok=True)

    pointer_data_paths = [
        str(pointer_memory_dir / f"{Path(path).name}.pt")
        for path in input_image_paths
    ]

    return input_image_paths, pointer_data_paths

def main():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet scenes for a specific GPU')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (0-indexed)')
    parser.add_argument('--total-gpus', type=int, required=True, help='Total number of GPUs')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    total_gpus = args.total_gpus

    # Get all paths
    input_image_paths, pointer_data_paths = setup_scannet_paths()

    # Split data across GPUs
    total_scenes = len(input_image_paths)
    scenes_per_gpu = (total_scenes + total_gpus - 1) // total_gpus
    start_idx = gpu_id * scenes_per_gpu
    end_idx = min(start_idx + scenes_per_gpu, total_scenes)

    # Get this GPU's subset
    local_input_paths = input_image_paths[start_idx:end_idx]
    local_output_paths = pointer_data_paths[start_idx:end_idx]

    print(f"GPU {gpu_id}/{total_gpus-1}: Processing {len(local_input_paths)} scenes (indices {start_idx} to {end_idx-1})")
    print(f"Total scenes in dataset: {total_scenes}")

    # Load models - will use CUDA_VISIBLE_DEVICES=X so it sees only one GPU as cuda:0
    model, processor, min_pixels, max_pixels, point3r_model = load_models(device=None)

    # Process this GPU's subset with progress bar
    for input_images_dir, pointer_data_path in tqdm(
        zip(local_input_paths, local_output_paths),
        desc=f"GPU {gpu_id}",
        total=len(local_input_paths)
    ):
        # Skip if already processed
        if Path(pointer_data_path).exists():
            continue

        preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
                          input_images_dir, pointer_data_path, use_viser=False, unload_point3r_model=False)

    print(f"\nGPU {gpu_id}: Completed processing {len(local_input_paths)} scenes!")

if __name__ == "__main__":
    main()
