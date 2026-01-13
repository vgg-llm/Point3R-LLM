"""
Function to extract pointer memory from image inputs using Point3R model.

This module provides utilities to convert image inputs (from qwen_vl_utils)
into Point3R memory features that can be used with the Point3R-enhanced model.
"""

import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
import sys

from .inference import inference, get_pred_pts3d
from .point3r import LocalMemory
from .utils.geometry import geotrf
import viser


def prepare_images_for_point3r(image_inputs, target_size=(640, 480), crop_border=20):
    """
    Prepare images for Point3R processing.

    This function processes images similar to the ScanNetDataset pattern:
    - Crops borders if specified
    - Resizes to target dimensions
    - Converts to normalized tensors

    Args:
        image_inputs: List of images (PIL Images, numpy arrays, or file paths)
        target_size: Tuple of (width, height) for resizing (default: (640, 480))
        crop_border: Number of pixels to crop from each edge (default: 20)

    Returns:
        list: List of view dictionaries containing:
            - 'img': Normalized image tensor (3, H, W)
            - 'true_shape': Tensor of shape (2,) with [height, width]
            - 'img_mask': Boolean tensor indicating valid image
    """
    views = []

    for img_input in image_inputs:
        # Convert to PIL Image if needed
        if isinstance(img_input, str):
            image = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            image = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            image = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_input)}")

        # Crop borders if specified
        if crop_border > 0:
            image = ImageOps.crop(image, border=crop_border)

        # Resize to target dimensions
        image = image.resize(target_size, Image.LANCZOS)

        # Convert to tensor [0, 1] range (matching reference implementation)
        img_tensor = tvf.ToTensor()(image)  # Shape: (3, H, W), range [0, 1]

        # Add batch dimension to match Point3R's expectation
        # Point3R expects: (batch_size, 3, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

        # Create true_shape tensor [height, width] with batch dimension
        true_shape = torch.tensor([[image.height, image.width]], dtype=torch.int32)  # Shape: (1, 2)

        # Create img_mask with batch dimension
        img_mask = torch.tensor([True], dtype=torch.bool)  # Shape: (1,)

        # Create view dictionary
        view = {
            "img": img_tensor,
            "true_shape": true_shape,
            "img_mask": img_mask,
        }
        views.append(view)

    return views


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
    use_viser=False,
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
            - 'camera_poses': (Optional) Tensor of shape (num_frames, 7)
                             Camera poses for each frame in format [tx, ty, tz, qw, qx, qy, qz]
                             Only present if the Point3R model has pose_head=True
                             Translation: [tx, ty, tz] - absolute position in 3D space
                             Rotation: [qw, qx, qy, qz] - unit quaternion (real part first)
                             Coordinate convention: OpenCV camera-to-world transformation

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

    # Prepare images for Point3R using the simplified function
    # Determine target size based on the size parameter
    if size == 512:
        target_size = (640, 480)  # Default for size=512
    elif size == 224:
        target_size = (224, 224)
    else:
        # For other sizes, maintain 4:3 aspect ratio
        target_size = (size, int(size * 3 / 4))

    crop_border = 0 if no_crop else 20
    views = prepare_images_for_point3r(
        image_inputs,
        target_size=target_size,
        crop_border=crop_border
    )

    if verbose:
        for i, view in enumerate(views):
            print(f"Processed image {i+1}/{len(image_inputs)}: shape {view['true_shape']}")

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

    if use_viser:
        server = viser.ViserServer()
        colors = []
        pts_3ds = []
        confs = []

        for idx, (pred, view) in enumerate(zip(outputs['pred'], outputs['views'])):
            pts_3d = get_pred_pts3d(None, pred, use_pose=True)
            color = view['img'].permute(0, 2, 3, 1)
            conf = pred['conf']

            color = color.detach().cpu().numpy() * 255
            color = color.astype(np.uint8)
            pts_3d = pts_3d.detach().cpu().numpy().reshape(-1, 3)
            color = color.reshape(-1, 3)
            conf = conf.detach().cpu().numpy().reshape(-1)

            colors.append(color)
            pts_3ds.append(pts_3d)
            confs.append(conf)

        colors = np.concatenate(colors, axis=0)
        pts_3ds = np.concatenate(pts_3ds, axis=0)
        confs = np.concatenate(confs, axis=0)

        quantile = np.quantile(confs, 0.10)
        mask = confs > quantile
        pts_3ds = pts_3ds[mask]
        colors = colors[mask]

        center = np.mean(pts_3ds, axis=0, keepdims=True)
        pts_3ds = pts_3ds - center

        server.scene.add_point_cloud(
            name=f"cloud",
            points=pts_3ds,
            colors=colors,
            point_size=0.001,
            visible=False
        )
        input("Press Enter to move on...")
        
    # Extract memory_aligned_image_embeds from Point3R outputs
    # This is now returned by Point3R's forward pass
    if 'memory_aligned_image_embeds' in outputs and outputs['memory_aligned_image_embeds'] is not None:
        memory_aligned_image_embeds = outputs['memory_aligned_image_embeds']

        # Handle list format (from merge mode with variable lengths)
        if isinstance(memory_aligned_image_embeds, list):
            # For demo, we typically use the last frame's embeddings
            # Or concatenate all frames
            print(f'{len(memory_aligned_image_embeds)} samples')
            memory_aligned_image_embeds = memory_aligned_image_embeds[-1]  # Take last sample's
            # Alternative: memory_aligned_image_embeds = torch.cat(memory_aligned_image_embeds, dim=1)

        # Ensure shape is (num_patches, 2048)
        if memory_aligned_image_embeds.dim() == 3:
            print(f'shape: {memory_aligned_image_embeds.shape}')
            # Shape: (bs, num_patches, 2048) → (num_patches, 2048)
            memory_aligned_image_embeds = memory_aligned_image_embeds[0]

        # Already at 2048-dim (Qwen's native dimension) - no projection needed
        pointer_memory_embeds = memory_aligned_image_embeds  # (num_patches, 2048)

        if verbose:
            print(f"Extracted memory_aligned_image_embeds from Point3R: {memory_aligned_image_embeds.shape}")
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

            # # Get min/max for each dimension
            # x_min, x_max = xyz_positions[:, 0].min(), xyz_positions[:, 0].max()
            # y_min, y_max = xyz_positions[:, 1].min(), xyz_positions[:, 1].max()
            # z_min, z_max = xyz_positions[:, 2].min(), xyz_positions[:, 2].max()

            # # Quantize each dimension to 0-32 range
            # if x_max > x_min:
            #     x_quantized = ((xyz_positions[:, 0] - x_min) / (x_max - x_min) * 32).long().clamp(0, 32)
            # else:
            #     x_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            # if y_max > y_min:
            #     y_quantized = ((xyz_positions[:, 1] - y_min) / (y_max - y_min) * 32).long().clamp(0, 32)
            # else:
            #     y_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            # if z_max > z_min:
            #     z_quantized = ((xyz_positions[:, 2] - z_min) / (z_max - z_min) * 32).long().clamp(0, 32)
            # else:
            #     z_quantized = torch.zeros(xyz_positions.shape[0], dtype=torch.long)

            # # Overwrite pointer_positions with quantized xyz values
            # pointer_positions = torch.stack([x_quantized, y_quantized, z_quantized], dim=1)

            pointer_positions = xyz_positions

            if verbose:
                print(f"Using pos_decode_memory from Point3R outputs")
                print(f"  - Number of memory points: {xyz_positions.shape[0]}")
                print(f"  - World xyz ranges: x[{xyz_positions[:, 0].min():.3f}, {xyz_positions[:, 0].max():.3f}], "
                      f"y[{xyz_positions[:, 1].min():.3f}, {xyz_positions[:, 1].max():.3f}], "
                      f"z[{xyz_positions[:, 2].min():.3f}, {xyz_positions[:, 2].max():.3f}]")

    # Extract camera poses from Point3R predictions (if pose_head=True)
    camera_poses = []
    if 'pred' in outputs and outputs['pred'] is not None:
        for i, pred in enumerate(outputs['pred']):
            if 'camera_pose' in pred and pred['camera_pose'] is not None:
                # camera_pose shape: (batch_size, 7) where 7 = [tx, ty, tz, qw, qx, qy, qz]
                pose = pred['camera_pose']
                if pose.dim() == 2:
                    # Take first batch element if batched
                    pose = pose[0]  # Shape: (7,)
                camera_poses.append(pose.cpu())

        if len(camera_poses) > 0:
            # Stack all camera poses: (num_frames, 7)
            camera_poses = torch.stack(camera_poses, dim=0)

            if verbose:
                print(f"Extracted camera poses:")
                print(f"  - Number of frames: {camera_poses.shape[0]}")
                print(f"  - Pose format: [tx, ty, tz, qw, qx, qy, qz]")
                print(f"  - Translation ranges: x[{camera_poses[:, 0].min():.3f}, {camera_poses[:, 0].max():.3f}], "
                      f"y[{camera_poses[:, 1].min():.3f}, {camera_poses[:, 1].max():.3f}], "
                      f"z[{camera_poses[:, 2].min():.3f}, {camera_poses[:, 2].max():.3f}]")

                # Print first camera pose
                first_pose = camera_poses[0]
                print(f"  - First camera pose:")
                print(f"    Translation: [{first_pose[0]:.4f}, {first_pose[1]:.4f}, {first_pose[2]:.4f}]")
                print(f"    Rotation (quat): [{first_pose[3]:.4f}, {first_pose[4]:.4f}, {first_pose[5]:.4f}, {first_pose[6]:.4f}]")

                # Print last camera pose if there's more than one frame
                if camera_poses.shape[0] > 1:
                    last_pose = camera_poses[-1]
                    print(f"  - Last camera pose:")
                    print(f"    Translation: [{last_pose[0]:.4f}, {last_pose[1]:.4f}, {last_pose[2]:.4f}]")
                    print(f"    Rotation (quat): [{last_pose[3]:.4f}, {last_pose[4]:.4f}, {last_pose[5]:.4f}, {last_pose[6]:.4f}]")

            # Transform pointer_positions using the first frame's camera pose
            if pointer_positions is not None:
                from .utils.camera import pose_encoding_to_camera

                # Convert first frame's camera pose (c2w) to camera matrix
                first_pose = camera_poses[0:1]  # Shape: (1, 7)
                c2w_matrix = pose_encoding_to_camera(first_pose, pose_encoding_type='absT_quaR')  # Shape: (1, 4, 4)
                c2w_matrix = c2w_matrix[0]  # Shape: (4, 4)

                # We want to transform world coordinates to first camera coordinates
                # So we need the inverse: w2c = inv(c2w)
                w2c_matrix = torch.inverse(c2w_matrix)  # Shape: (4, 4)

                # pointer_positions is continuous xyz from pos_decode_memory
                xyz_world = pointer_positions.float()  # Shape: (num_points, 3)

                # Convert to homogeneous coordinates
                ones = torch.ones(xyz_world.shape[0], 1)
                xyz_world_homogeneous = torch.cat([xyz_world, ones], dim=1)  # Shape: (num_points, 4)

                # Transform to first camera frame
                xyz_cam_homogeneous = (w2c_matrix @ xyz_world_homogeneous.T).T  # Shape: (num_points, 4)
                pointer_positions = xyz_cam_homogeneous[:, :3]  # Shape: (num_points, 3)

                if verbose:
                    print(f"Transformed pointer positions to first camera frame")
                    print(f"  - Camera xyz ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                          f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                          f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")
        else:
            camera_poses = None
            if verbose:
                print(f"No camera poses found (pose_head may be disabled)")
    else:
        camera_poses = None

    if verbose:
        print(f"Extracted pointer memory:")
        print(f"  - Number of pointers: {pointer_memory_embeds.shape[0]}")
        print(f"  - Memory embeddings shape: {pointer_memory_embeds.shape}")
        print(f"  - Pointer positions shape: {pointer_positions.shape}")
        if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
            print(f"  - Final position ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                  f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                  f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")

    result = {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions': pointer_positions,
    }

    # Add camera poses if available
    if camera_poses is not None:
        result['camera_poses'] = camera_poses

    return result


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

    # Access camera poses (if pose_head=True)
    if 'camera_poses' in pointer_data:
        camera_poses = pointer_data['camera_poses']  # Shape: (num_frames, 7)
        # Each pose: [tx, ty, tz, qw, qx, qy, qz]

        # Convert to 4x4 camera-to-world matrices
        from src.qwen_vl.model.point3r.utils.camera import pose_encoding_to_camera
        c2w_matrices = pose_encoding_to_camera(camera_poses, pose_encoding_type='absT_quaR')
        # Shape: (num_frames, 4, 4) - OpenCV camera-to-world transformations
    """)
