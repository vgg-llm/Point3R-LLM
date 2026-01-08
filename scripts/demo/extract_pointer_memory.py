"""
Function to extract pointer memory from image inputs using Point3R model.

This module provides utilities to convert image inputs (from qwen_vl_utils)
into Point3R memory features that can be used with the Point3R-enhanced model.
"""

import torch
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from qwen_vl.model.point3r.inference import inference
from qwen_vl.model.point3r.point3r import LocalMemory

# Image normalization (same as Point3R)
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    """Resize PIL image maintaining aspect ratio."""
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def prepare_image_for_point3r(image, size=512, crop=True, square_ok=False):
    """
    Prepare a single image for Point3R processing.

    Args:
        image: PIL Image, numpy array, or file path
        size: Target size for the longer edge (default: 512)
        crop: Whether to crop or resize to target dimensions
        square_ok: Whether square images are acceptable

    Returns:
        dict: Dictionary containing processed image tensor and metadata
    """
    # Convert to PIL Image if needed
    if isinstance(image, str):
        image = exif_transpose(Image.open(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Get original size
    W1, H1 = image.size

    # Resize
    if size == 224:
        # resize short side to 224 (then crop)
        image = _resize_pil_image(image, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to target size
        image = _resize_pil_image(image, size)

    W, H = image.size
    cx, cy = W // 2, H // 2

    # Crop or resize to final dimensions
    if size == 224:
        half = min(cx, cy)
        if crop:
            image = image.crop((cx - half, cy - half, cx + half, cy + half))
        else:  # resize
            image = image.resize((2 * half, 2 * half), Image.LANCZOS)
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = 3 * halfw // 4
        if crop:
            image = image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        else:  # resize
            image = image.resize((2 * halfw, 2 * halfh), Image.LANCZOS)

    W2, H2 = image.size

    # Return processed image
    return {
        "img": ImgNorm(image)[None],  # Add batch dimension
        "true_shape": np.int32([H2, W2]),  # Note: height, width order
    }


def extract_pointer_memory(
    image_inputs,
    point3r_model,
    image_embeds=None,
    grid_thw=None,
    device='cuda',
    no_crop=False,
    full_seq=False,
    size=512,
    verbose=True,
):
    """
    Extract pointer memory from image inputs using Point3R model.

    This function processes images through Point3R to generate memory features
    that can be used with Qwen2_5_VLForConditionalGenerationWithPoint3R.

    Args:
        image_inputs: List of image inputs (can be PIL Images, file paths, or numpy arrays)
        point3r_model: Initialized Point3R model
        device: Device to run inference on (default: 'cuda')
        no_crop: If True, resize instead of crop (default: False)
        full_seq: If True, process full sequence mode (default: False)
        size: Target image size (default: 512)
        verbose: Print progress information (default: True)

    Returns:
        dict: Dictionary containing:
            - 'pointer_memory_embeds': Tensor of shape (num_pointers, hidden_dim)
                                      Memory embeddings for each pointer token
            - 'pointer_positions': Tensor of shape (num_pointers, 3)
                                  3D positions (height, width, depth) for each pointer
            - 'pts3d': Raw 3D point cloud from Point3R (for debugging)
            - 'metadata': Additional metadata from Point3R processing

    Example:
        >>> from qwen_vl_utils import process_vision_info
        >>> messages = [{"role": "user", "content": [{"type": "image", "image": "path/to/image.jpg"}]}]
        >>> image_inputs, _ = process_vision_info(messages)
        >>>
        >>> pointer_data = extract_pointer_memory(
        ...     image_inputs,
        ...     point3r_model,
        ...     device='cuda'
        ... )
        >>>
        >>> # Use with Qwen2_5_VLForConditionalGenerationWithPoint3R
        >>> inputs = processor(
        ...     text=["<|pointer_pad|> What's in this scene?"],
        ...     return_tensors="pt",
        ... )
        >>> outputs = model.generate(
        ...     **inputs,
        ...     pointer_memory_embeds=pointer_data['pointer_memory_embeds'],
        ...     pointer_positions=pointer_data['pointer_positions'],
        ... )
    """

    # Ensure image_inputs is a list
    if not isinstance(image_inputs, list):
        image_inputs = [image_inputs]

    # Prepare images for Point3R
    views = []
    for i, img_input in enumerate(image_inputs):
        # Process each image
        processed = prepare_image_for_point3r(
            img_input,
            size=size,
            crop=not no_crop,
        )

        # Create view dict matching Point3R's expected format
        view = {
            "img": processed["img"],
            "ray_map": torch.full(
                (
                    processed["img"].shape[0],
                    6,
                    processed["img"].shape[-2],
                    processed["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(processed["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)

        if verbose:
            print(f"Processed image {i+1}/{len(image_inputs)}: shape {processed['true_shape']}")

    # Run Point3R inference
    if verbose:
        print(f"Running Point3R inference on {len(views)} image(s)...")

    outputs = inference(
        views,
        point3r_model,
        device,
        image_embeds=image_embeds,
        grid_thw=grid_thw,
        verbose=verbose
    )

    # Extract pointer_aligned_image_embeds from Point3R outputs
    # This is now returned by Point3R's forward pass
    if 'pointer_aligned_image_embeds' in outputs and outputs['pointer_aligned_image_embeds'] is not None:
        pointer_aligned_image_embeds = outputs['pointer_aligned_image_embeds']

        # Handle list format (from merge mode with variable lengths)
        if isinstance(pointer_aligned_image_embeds, list):
            # For demo, we typically use the last frame's embeddings
            # Or concatenate all frames
            print(f'{len(pointer_aligned_image_embeds)} samples')
            pointer_aligned_image_embeds = pointer_aligned_image_embeds[-1]  # Take last sample's
            # Alternative: pointer_aligned_image_embeds = torch.cat(pointer_aligned_image_embeds, dim=1)

        # Ensure shape is (num_patches, 2048)
        if pointer_aligned_image_embeds.dim() == 3:
            print(f'shape: {pointer_aligned_image_embeds.shape}')
            # Shape: (bs, num_patches, 2048) → (num_patches, 2048)
            pointer_aligned_image_embeds = pointer_aligned_image_embeds[0]

        # Already at 2048-dim (Qwen's native dimension) - no projection needed
        pointer_memory_embeds = pointer_aligned_image_embeds  # (num_patches, 2048)

        if verbose:
            print(f"Extracted pointer_aligned_image_embeds from Point3R: {pointer_aligned_image_embeds.shape}")
    else:
        raise ValueError

    # Use pos_decode_memory from Point3R outputs if available (for merged memory)
    if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
        pos_decode_memory = outputs['pos_decode_memory']
        
        # Handle list format (from merge mode with variable lengths)
        if isinstance(pos_decode_memory, list):
            pos_decode_memory = pos_decode_memory[-1]  # Take last sample's
            # Alternative: pos_decode_memory = torch.cat(pos_decode_memory, dim=1)

        # Handle list format (from merge mode with variable lengths per batch)
        if pos_decode_memory.dim() == 3:
            # Shape: (bs, num_patches, 3) → (num_patches, 3)
            pos_decode_memory = pos_decode_memory[0]

        if pos_decode_memory is not None:
            # pos_decode_memory shape: (num_memory_points, 3) where 3 is (x, y, z)
            # Quantize xyz values to integers from 0 to 32
            xyz_positions = pos_decode_memory.cpu()

            # Get min/max for each dimension
            x_min, x_max = xyz_positions[:, 0].min(), xyz_positions[:, 0].max()
            y_min, y_max = xyz_positions[:, 1].min(), xyz_positions[:, 1].max()
            z_min, z_max = xyz_positions[:, 2].min(), xyz_positions[:, 2].max()

            # Quantize each dimension to 0-32 range
            if x_max > x_min:
                x_quantized = ((xyz_positions[:, 0] - x_min) / (x_max - x_min) * 32).long().clamp(0, 32)
            else:
                x_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            if y_max > y_min:
                y_quantized = ((xyz_positions[:, 1] - y_min) / (y_max - y_min) * 32).long().clamp(0, 32)
            else:
                y_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            if z_max > z_min:
                z_quantized = ((xyz_positions[:, 2] - z_min) / (z_max - z_min) * 32).long().clamp(0, 32)
            else:
                z_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            # Overwrite pointer_positions with quantized xyz values
            pointer_positions = torch.stack([x_quantized, y_quantized, z_quantized], dim=1)

            if verbose:
                print(f"Using pos_decode_memory from Point3R outputs")
                print(f"  - Number of memory points: {xyz_positions.shape[0]}")
                print(f"  - Original xyz ranges: x[{x_min:.3f}, {x_max:.3f}], "
                      f"y[{y_min:.3f}, {y_max:.3f}], z[{z_min:.3f}, {z_max:.3f}]")
                print(f"  - Quantized to [0, 32] range")

    if verbose:
        print(f"Extracted pointer memory:")
        print(f"  - Number of pointers: {pointer_memory_embeds.shape[0]}")
        print(f"  - Memory embeddings shape: {pointer_memory_embeds.shape}")
        print(f"  - Pointer positions shape: {pointer_positions.shape}")
        if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
            print(f"  - Position ranges: x[{pointer_positions[:, 0].min()}-{pointer_positions[:, 0].max()}], "
                  f"y[{pointer_positions[:, 1].min()}-{pointer_positions[:, 1].max()}], "
                  f"z[{pointer_positions[:, 2].min()}-{pointer_positions[:, 2].max()}]")

    return {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions': pointer_positions,
    }


if __name__ == "__main__":
    # Example usage
    print("Example: Extract pointer memory from an image")
    print("=" * 70)

    # This is a demonstration - you would provide actual images
    print("Usage:")
    print("""
    from extract_pointer_memory import extract_pointer_memory
    from qwen_vl.model.point3r.point3r import Point3R

    # Load Point3R model
    point3r_model = Point3R.from_pretrained("path/to/point3r_checkpoint.pth")
    point3r_model = point3r_model.to('cuda')
    point3r_model.eval()

    # Extract memory from images
    pointer_data = extract_pointer_memory(
        image_inputs=['path/to/image1.jpg', 'path/to/image2.jpg'],
        point3r_model=point3r_model,
        device='cuda',
        no_crop=False,
        size=512,
    )

    # Use with Qwen2.5-VL model
    outputs = model.generate(
        **inputs,
        pointer_memory_embeds=pointer_data['pointer_memory_embeds'],
        pointer_positions=pointer_data['pointer_positions'],
    )
    """)
