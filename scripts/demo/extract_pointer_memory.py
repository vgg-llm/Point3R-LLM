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
import os

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

    # Extract 3D points from predictions
    pts3ds = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]

    if verbose:
        print(f"Point3R inference complete. Extracted {len(pts3ds)} point clouds.")

    # For single image, use the first prediction
    # For multiple images, you might want to aggregate or use specific views
    pts3d = pts3ds[0] if len(pts3ds) > 0 else None

    if pts3d is None:
        raise ValueError("Point3R inference did not produce valid 3D points")

    # Extract memory features and positions
    # pts3d shape: (batch_size, height, width, 3) where last dim is (x, y, z)
    bs, img_h, img_w, _ = pts3d.shape

    # Downsample to patch-level positions (matching Point3R's 16x16 patches)
    img_pos_len_h = img_h // 16
    img_pos_len_w = img_w // 16

    # Average 3D positions over each 16x16 patch
    img_pos = pts3d.permute(0, 3, 1, 2)  # (bs, 3, h, w)
    img_pos = img_pos.unfold(2, 16, 16)  # (bs, 3, h_patches, w, 16)
    img_pos = img_pos.unfold(3, 16, 16)  # (bs, 3, h_patches, w_patches, 16, 16)
    img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1)
    img_pos = img_pos.permute(0, 2, 3, 1).reshape(bs, -1, 3)  # (bs, num_patches, 3)

    # For pointer positions: convert (x, y, z) to (height, width, depth) indices
    # Normalize coordinates to integer grid positions
    num_patches = img_pos.shape[1]

    # Create height and width grid indices
    h_indices = torch.arange(img_pos_len_h).view(-1, 1).expand(-1, img_pos_len_w).flatten()
    w_indices = torch.arange(img_pos_len_w).view(1, -1).expand(img_pos_len_h, -1).flatten()

    # Use z-coordinate (depth) directly, normalized to reasonable range
    z_coords = img_pos[0, :, 2]  # (num_patches,)

    # Normalize depth to integer indices (scale to a reasonable range, e.g., 0-100)
    z_min, z_max = z_coords.min(), z_coords.max()
    if z_max > z_min:
        d_indices = ((z_coords - z_min) / (z_max - z_min) * 100).long()
    else:
        d_indices = torch.zeros_like(z_coords).long()

    # Combine into pointer_positions (num_pointers, 3) where dims are (h, w, d)
    pointer_positions = torch.stack([h_indices, w_indices, d_indices], dim=1)

    # Extract pointer_aligned_image_embeds from Point3R outputs
    # This is now returned by Point3R's forward pass
    if 'pointer_aligned_image_embeds' in outputs and outputs['pointer_aligned_image_embeds'] is not None:
        pointer_aligned_image_embeds = outputs['pointer_aligned_image_embeds']

        # Handle list format (from merge mode with variable lengths)
        if isinstance(pointer_aligned_image_embeds, list):
            # For demo, we typically use the last frame's embeddings
            # Or concatenate all frames
            pointer_aligned_image_embeds = pointer_aligned_image_embeds[-1]  # Take last frame
            # Alternative: pointer_aligned_image_embeds = torch.cat(pointer_aligned_image_embeds, dim=1)

        # Ensure shape is (num_patches, 2048)
        if pointer_aligned_image_embeds.dim() == 3:
            # Shape: (bs, num_patches, 2048) → (num_patches, 2048)
            pointer_aligned_image_embeds = pointer_aligned_image_embeds[0]

        # Already at 2048-dim (Qwen's native dimension) - no projection needed
        pointer_memory_embeds = pointer_aligned_image_embeds  # (num_patches, 2048)

        if verbose:
            print(f"Extracted pointer_aligned_image_embeds from Point3R: {pointer_aligned_image_embeds.shape}")
    else:
        # Fallback to placeholder if Point3R didn't return it
        print("Warning: Point3R did not return pointer_aligned_image_embeds, using placeholder")

        # Get hidden dimension from the model (typically 2048 for Qwen2.5-VL-3B)
        hidden_dim = 2048  # Qwen2.5-VL-3B hidden size

        # Create simple position-encoded embeddings as placeholder
        pointer_memory_embeds = torch.zeros(num_patches, hidden_dim)

        # Encode 3D positions into the embeddings (simple encoding)
        position_encoding = torch.cat([
            img_pos[0].repeat(1, hidden_dim // 6)[:, :hidden_dim // 3],  # x, y, z positions
            torch.sin(img_pos[0] * 10).repeat(1, hidden_dim // 6)[:, :hidden_dim // 3],  # sin encoding
            torch.cos(img_pos[0] * 10).repeat(1, hidden_dim // 6)[:, :hidden_dim // 3],  # cos encoding
        ], dim=1)

        pointer_memory_embeds[:, :position_encoding.shape[1]] = position_encoding

    if verbose:
        print(f"Extracted pointer memory:")
        print(f"  - Number of pointers: {num_patches}")
        print(f"  - Memory embeddings shape: {pointer_memory_embeds.shape}")
        print(f"  - Pointer positions shape: {pointer_positions.shape}")
        print(f"  - Position ranges: h[{h_indices.min()}-{h_indices.max()}], "
              f"w[{w_indices.min()}-{w_indices.max()}], d[{d_indices.min()}-{d_indices.max()}]")

    return {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions': pointer_positions,
        'pts3d': pts3d,
        'metadata': {
            'num_views': len(views),
            'image_size': (img_h, img_w),
            'num_patches': (img_pos_len_h, img_pos_len_w),
            'depth_range': (z_min.item(), z_max.item()),
        }
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
