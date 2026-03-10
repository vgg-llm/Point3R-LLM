"""
Function to extract pointer memory from image inputs using Point3R model.

This module provides utilities to convert image inputs (from qwen_vl_utils)
into Point3R memory features that can be used with the Point3R-enhanced model.
"""

import json
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
import sys

from .inference import inference, get_pred_pts3d
from .point3r import LocalMemory
from .utils.geometry import geotrf
from typing import List, Dict


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


def visualize_point3r_viser(
    pointer_data,
    annotation_result=None,
    scannet_pth_path=None,
    scannet_pose_paths=None,
    attention_data=None,
    fast=False,
    point_stride=1,
):
    """
    Launch an interactive viser 3D visualization of Point3R outputs.

    Call this after extract_pointer_memory() with its return dict.

    Args:
        pointer_data: Return dict from extract_pointer_memory (must include '_point3r_outputs').
        annotation_result: Output from extract_box_and_coordinates_from_scan2cap for visualization.
        scannet_pth_path: Path to ScanNet .pth file for GT point cloud visualization.
        scannet_pose_paths: List of paths to ScanNet pose .txt files with camera poses.
        attention_data: Optional dict with attention visualization data containing:
            - "attention_matrix": Tensor (num_gen_tokens, num_pointers)
            - "generated_tokens_text": list[str]
            - "pointer_positions": np.ndarray (num_pointers, 3)
            - "pointer_timestamps": np.ndarray (num_pointers,)
    """
    import viser
    import viser.transforms as tf
    import matplotlib.cm as cm
    import time
    import threading

    # Fast mode defaults
    if fast and point_stride == 1:
        point_stride = 4
    if fast:
        print(f"Fast mode enabled: point_stride={point_stride}")

    outputs = pointer_data['_point3r_outputs']
    pointer_positions = pointer_data['pointer_positions']
    camera_poses = pointer_data.get('camera_poses')

    viser_start_time = time.time()
    server = viser.ViserServer()

    if annotation_result is not None:
        # Get transformation matrices from the first element (already computed as numpy arrays)
        first_elem = annotation_result['all_elements'][0]
        global2cam = first_elem['global2cam']
        ref_cam2global = first_elem['ref_cam2global']
        axis_align_matrix = first_elem['axis_align']

        # Visualize ScanNet pose files if provided
        if scannet_pose_paths is not None and len(scannet_pose_paths) > 0:
            print(f"Loading {len(scannet_pose_paths)} ScanNet pose files...")
            gt_frustums: List[viser.CameraFrustumHandle] = []
            for i, pose_path in enumerate(scannet_pose_paths):
                try:
                    # Load 4x4 pose matrix from txt file
                    pose_matrix = np.loadtxt(pose_path)
                    if pose_matrix.shape == (4, 4):
                        aligned_pose_matrix = axis_align_matrix @ pose_matrix
                        aligned_pose_se3 = tf.SE3.from_matrix(aligned_pose_matrix)
                        h, w = 480, 640
                        fy = 1.1 * h
                        fov = 2 * np.arctan2(h / 2, fy)
                        gt_frustum = server.scene.add_camera_frustum(
                            f"gt_camera_{i}",
                            fov=fov,
                            aspect=w / h,
                            scale=0.05,
                            image=None,
                            line_width=1.0,
                            color=(255, 165, 0),  # Orange color for GT poses
                            position=aligned_pose_se3.translation(),
                            wxyz=aligned_pose_se3.rotation().wxyz
                        )
                        gt_frustums.append(gt_frustum)
                    if i == 0:
                        ref_cam2global = pose_matrix
                        ref_cam2global_se3 = tf.SE3.from_matrix(ref_cam2global)
                        print('Overwrite ref_cam2global')
                        print(f"GT ref.frame camera pose: {pose_se3}")
                        print(f"Given ref.frame camera pose: {ref_cam2global_se3}")


                except Exception as e:
                    print(f"  Warning: Failed to load pose from {pose_path}: {e}")
            print(f"  Added {len(gt_frustums)} GT camera frustums (orange)")


        extrinsic = axis_align_matrix @ ref_cam2global
        extrinsic_se3 = tf.SE3.from_matrix(extrinsic)
        align_wxyz = extrinsic_se3.rotation().wxyz
        align_pos = extrinsic_se3.translation()
        print('Current extrinsic:', align_wxyz, align_pos, sep='\n')
    else:
        axis_align_matrix = np.eye(4)
        extrinsic_se3 = tf.SE3.from_matrix(np.eye(4))
        align_wxyz = extrinsic_se3.rotation().wxyz
        align_pos = extrinsic_se3.translation()
        print('No annotation given.')

    # Store per-frame data for timestamp visualization
    per_frame_data = []
    num_frames = len(outputs['pred'])

    for idx, (pred, view) in enumerate(zip(outputs['pred'], outputs['views'])):
        pts_3d = get_pred_pts3d(None, pred, use_pose=True)

        # Extract confidence (per-pixel) — use cross-view conf, fall back to conf_self
        if 'conf' in pred:
            conf = pred['conf'].detach().cpu().numpy().reshape(-1)
        elif 'conf_self' in pred:
            conf = pred['conf_self'].detach().cpu().numpy().reshape(-1)
        else:
            conf = None

        # Original RGB image for frustum display (H, W, 3)
        rgb_image = view['img'].permute(0, 2, 3, 1).squeeze(0)
        rgb_image_np = (rgb_image.detach().cpu().numpy() * 255).astype(np.uint8)

        # Points and colors (NO quantile filtering for interpretability)
        pts_3d_np = pts_3d.detach().cpu().numpy().reshape(-1, 3)
        color_rgb = rgb_image_np.reshape(-1, 3)

        # Spatial downsampling: subsample the 2D pixel grid
        if point_stride > 1:
            H, W = rgb_image_np.shape[:2]
            pts_3d_np = pts_3d_np.reshape(H, W, 3)[::point_stride, ::point_stride].reshape(-1, 3)
            color_rgb = color_rgb.reshape(H, W, 3)[::point_stride, ::point_stride].reshape(-1, 3)
            if conf is not None:
                conf = conf.reshape(H, W)[::point_stride, ::point_stride].reshape(-1)

        # Viridis color for this frame's points
        norm_idx = idx / max(num_frames - 1, 1)
        viridis_rgba = cm.viridis(norm_idx)
        color_timestamp = np.full_like(color_rgb, (np.array(viridis_rgba[:3]) * 255).astype(np.uint8))

        per_frame_data.append({
            'pts_3d': pts_3d_np,
            'colors_rgb': color_rgb,
            'colors_timestamp': color_timestamp,
            'rgb_image': rgb_image_np,
            'confidence': conf,
            'frame_idx': idx,
        })

    # === Pre-compute attention assignment (if attention_data provided) ===
    attn_assignments = None
    attn_threshold = None
    attn_initial_colors = None
    attn_initial_masks = None
    attn_initial_overlay = None
    if attention_data is not None and not fast:
        import sys
        sys.path.insert(0, 'scripts/demo')
        from attention_utils import (
            precompute_dense_point_assignment,
            compute_attention_colors_from_assignment,
            aggregate_attention_across_tokens,
        )

        attn_ptr_pos = attention_data['pointer_positions']
        attn_ptr_ts = attention_data['pointer_timestamps']
        per_frame_pts3d_list = [fd['pts_3d'] for fd in per_frame_data]

        print("Pre-computing attention assignment (KD-tree per frame)...")
        attn_assignments, attn_threshold, attn_sigma = precompute_dense_point_assignment(
            per_frame_pts3d_list, attn_ptr_pos, attn_ptr_ts
        )
        print(f"  Threshold: {attn_threshold:.4f}, Sigma: {attn_sigma:.4f}")

        # Compute initial attention colors (aggregated mean)
        init_weights = aggregate_attention_across_tokens(
            attention_data['attention_matrix'], mode="mean"
        ).numpy()
        attn_initial_colors, attn_initial_masks, _ = compute_attention_colors_from_assignment(
            attn_assignments, attn_threshold, init_weights, per_frame_pts3d_list,
            sigma=attn_sigma * 5.0, gamma=1.5,
        )
        # Compute inferno attention colors (for overlay blending)
        attn_initial_viridis, _, attn_initial_alpha = compute_attention_colors_from_assignment(
            attn_assignments, attn_threshold, init_weights, per_frame_pts3d_list,
            sigma=attn_sigma * 5.0, gamma=1.5, colormap="inferno",
        )
        attn_initial_overlay = [
            ((1 - (0.7 + 0.3 * a)[:, None]) * fd['colors_rgb'].astype(np.float32)
             + ((0.7 + 0.3 * a)[:, None]) * v.astype(np.float32)).astype(np.uint8)
            for fd, v, a in zip(per_frame_data, attn_initial_viridis, attn_initial_alpha)
        ]
    elif attention_data is not None and fast:
        print("  Skipping attention pre-computation (fast mode)")

    # === GUI Controls (skipped in fast mode) ===
    gui_conf_threshold = None
    gui_attn_token_slider = None
    gui_attn_token_label = None
    gui_attn_aggregation = None
    gui_attn_hide_unassigned = None
    gui_attn_gamma = None

    if not fast:
        with server.gui.add_folder("Playback"):
            gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames-1, step=1, initial_value=num_frames-1)
            gui_next_frame = server.gui.add_button("Next Frame")
            gui_prev_frame = server.gui.add_button("Prev Frame")
            gui_playing = server.gui.add_checkbox("Playing", False)
            gui_framerate = server.gui.add_slider("FPS", min=0.5, max=100, step=0.5, initial_value=1)
            gui_accumulative = server.gui.add_checkbox("Accumulative Mode", True)
            gui_stride = server.gui.add_slider("Stride", min=1, max=max(num_frames, 1), step=1, initial_value=1)
            gui_num_frames_visible = server.gui.add_slider("Num Frames Visible", min=1, max=max(num_frames, 1), step=1, initial_value=max(num_frames, 1))

        color_mode_options = ["Original RGB", "Timestamp (viridis)"]
        if attention_data is not None and attn_assignments is not None:
            color_mode_options.append("Attention Heatmap")
            color_mode_options.append("Attention Heatmap (overlay)")

        # Compute confidence stats for slider range
        has_confidence = any(fd['confidence'] is not None for fd in per_frame_data)
        if has_confidence:
            all_conf = np.concatenate([fd['confidence'] for fd in per_frame_data if fd['confidence'] is not None])
            conf_min, conf_max = float(all_conf.min()), float(all_conf.max())
            conf_median = float(np.median(all_conf))
            print(f"  Confidence range: [{conf_min:.2f}, {conf_max:.2f}], median: {conf_median:.2f}")
        else:
            conf_min, conf_max, conf_median = 0.0, 1.0, 0.5

        with server.gui.add_folder("Visualization"):
            default_color_mode = "Attention Heatmap (overlay)" if attn_initial_overlay is not None else "Original RGB"
            gui_color_mode = server.gui.add_dropdown("Color Mode", options=color_mode_options, initial_value=default_color_mode)
            gui_show_frustums = server.gui.add_checkbox("Show Frustums", False)
            gui_point_size = server.gui.add_slider("Point Size", min=0.0001, max=0.03, step=0.0005, initial_value=0.005)
            gui_frustum_scale = server.gui.add_slider("Frustum Scale", min=0.01, max=0.2, step=0.01, initial_value=0.05)
            if has_confidence:
                gui_conf_threshold = server.gui.add_slider(
                    "Confidence Threshold", min=0.9, max=conf_max,
                    step=(conf_max - 0.9) / 100.0, initial_value=1.10
                )

        if attention_data is not None and attn_assignments is not None:
            num_gen_tokens = attention_data['attention_matrix'].shape[0]
            with server.gui.add_folder("Attention Controls"):
                gui_attn_token_slider = server.gui.add_slider(
                    "Generated Token", min=0, max=num_gen_tokens,
                    step=1, initial_value=0,
                )
                gui_attn_token_label = server.gui.add_text(
                    "Token Text", initial_value="[Aggregated Mean]", disabled=True
                )
                gui_attn_aggregation = server.gui.add_dropdown(
                    "Aggregation Mode", options=["mean", "max", "sum"], initial_value="mean"
                )
                gui_attn_hide_unassigned = server.gui.add_checkbox(
                    "Hide Unassigned Points", initial_value=False
                )
                gui_attn_gamma = server.gui.add_slider(
                    "Gamma", min=0.1, max=2.0, step=0.1, initial_value=1.5
                )
                gui_attn_sigma_factor = server.gui.add_slider(
                    "Sigma Factor", min=0.1, max=30.0, step=0.1, initial_value=5.0
                )
                gui_overlay_blend = server.gui.add_slider(
                    "Overlay Blend", min=0.0, max=1.0, step=0.05, initial_value=1.0
                )

        # === Camera save/load ===
        _saved_camera: Dict = {}

        with server.gui.add_folder("Camera"):
            gui_save_cam = server.gui.add_button("Save Camera")
            gui_load_cam = server.gui.add_button("Load Camera")

        @gui_save_cam.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            _saved_camera['position'] = client.camera.position.tolist()
            _saved_camera['wxyz'] = client.camera.wxyz.tolist()
            _saved_camera['look_at'] = client.camera.look_at.tolist()
            _saved_camera['up_direction'] = client.camera.up_direction.tolist()
            with open("camera_state.json", "w") as f:
                json.dump(_saved_camera, f)
            print("Camera state saved.")

        @gui_load_cam.on_click
        def _(event: viser.GuiEvent):
            nonlocal _saved_camera
            client = event.client
            if not _saved_camera:
                try:
                    with open("camera_state.json") as f:
                        _saved_camera = json.load(f)
                except FileNotFoundError:
                    print("No saved camera state found.")
                    return
            client.camera.position = np.array(_saved_camera['position'])
            client.camera.wxyz = np.array(_saved_camera['wxyz'])
            client.camera.look_at = np.array(_saved_camera['look_at'])
            client.camera.up_direction = np.array(_saved_camera['up_direction'])
            print("Camera state loaded.")

    def get_conf_mask(frame_data):
        """Return boolean mask for points passing the confidence threshold."""
        conf = frame_data['confidence']
        if conf is None or gui_conf_threshold is None:
            return None
        return conf >= gui_conf_threshold.value

    def apply_mask(pts, colors, mask):
        """Apply confidence mask, returning at least 1 point for viser."""
        if mask is None:
            return pts, colors
        p, c = pts[mask], colors[mask]
        if len(p) == 0:
            return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8)
        return p, c

    # Create parent frame for all timesteps
    server.scene.add_frame("/frames", show_axes=False, wxyz=align_wxyz, position=align_pos)

    frame_nodes: List[viser.FrameHandle] = []
    point_cloud_handles: Dict[int, any] = {}  # single handle per frame
    frame_color_data: Dict[int, Dict[str, np.ndarray]] = {}  # cached color arrays
    frame_points_data: Dict[int, Dict[str, np.ndarray]] = {}  # cached point arrays (per mode, for hide_unassigned)
    frustum_handles: List[viser.CameraFrustumHandle] = []

    _default_point_size = 0.005

    def _color_mode_key():
        """Map GUI color mode string to dict key."""
        if fast:
            return 'rgb'
        mode = gui_color_mode.value
        if mode == "Timestamp (viridis)":
            return 'timestamp'
        elif mode == "Attention Heatmap":
            return 'attention'
        elif mode == "Attention Heatmap (overlay)":
            return 'attention_overlay'
        return 'rgb'

    for frame_data in per_frame_data:
        idx = frame_data['frame_idx']

        # Frame node for this timestep
        frame_node = server.scene.add_frame(f"/frames/t{idx}", show_axes=False)
        frame_nodes.append(frame_node)

        conf_mask = get_conf_mask(frame_data)
        pts_masked, rgb_masked = apply_mask(frame_data['pts_3d'], frame_data['colors_rgb'], conf_mask)
        _, ts_masked = apply_mask(frame_data['pts_3d'], frame_data['colors_timestamp'], conf_mask)

        # Cache all color and point variants
        color_cache = {'rgb': rgb_masked, 'timestamp': ts_masked}
        points_cache = {'rgb': pts_masked, 'timestamp': pts_masked}
        if attn_initial_colors is not None:
            _, attn_masked = apply_mask(frame_data['pts_3d'], attn_initial_colors[idx], conf_mask)
            color_cache['attention'] = attn_masked
            points_cache['attention'] = pts_masked
        if attn_initial_overlay is not None:
            _, overlay_masked = apply_mask(frame_data['pts_3d'], attn_initial_overlay[idx], conf_mask)
            _, viridis_masked = apply_mask(frame_data['pts_3d'], attn_initial_viridis[idx], conf_mask)
            _, alpha_masked = apply_mask(frame_data['pts_3d'], attn_initial_alpha[idx], conf_mask)
            color_cache['attention_overlay'] = overlay_masked
            color_cache['attention_viridis'] = viridis_masked
            color_cache['attention_rgb'] = rgb_masked
            color_cache['attention_alpha'] = alpha_masked
            points_cache['attention_overlay'] = pts_masked
        frame_color_data[idx] = color_cache
        frame_points_data[idx] = points_cache

        # Single point cloud per frame
        initial_colors = color_cache.get('attention_overlay', rgb_masked)
        pc = server.scene.add_point_cloud(
            name=f"/frames/t{idx}/points",
            points=pts_masked,
            colors=initial_colors,
            point_size=gui_point_size.value if not fast else _default_point_size,
            point_shape="rounded",
            visible=True,
        )
        point_cloud_handles[idx] = pc

        # Camera frustum with RGB image and viridis-colored edge
        if camera_poses is not None:
            pose = camera_poses[idx].numpy()
            pose_se3 = extrinsic_se3 @ tf.SE3(np.concatenate([pose[3:], pose[:3]]))

            norm_idx = idx / max(num_frames - 1, 1)
            frustum_color = tuple((np.array(cm.viridis(norm_idx)[:3]) * 255).astype(int))

            h, w = frame_data['rgb_image'].shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = server.scene.add_camera_frustum(
                f"/frames/t{idx}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05 if fast else gui_frustum_scale.value,
                image=None if fast else frame_data['rgb_image'],
                line_width=2.0,
                color=frustum_color,
                position=pose_se3.translation(),
                wxyz=pose_se3.rotation().wxyz,
                visible=False,
            )
            frustum_handles.append(frustum)


    server.scene.add_point_cloud(
        name=f"pointer_memory_anchor",
        points=pointer_positions.numpy(),
        colors=(255, 0, 0),
        point_size=0.02,
        visible=False,
        wxyz=align_wxyz,
        position=align_pos

    )

    # Visualize annotation data if provided
    point_array = np.array([[0, 0, 0]])
    if annotation_result is not None:
        # Get all elements from annotation result
        all_elements = annotation_result.get('all_elements', [])
        framewise_elements = annotation_result.get('by_pointer_data')
        for elem in all_elements:
            obj_id = elem.get('metadata', {}).get('object_id', 'unknown')
            box_center = elem.get('box_center')
            transformed_box_center = elem.get('transformed_center')
            if transformed_box_center is not None:
                pos = tuple(transformed_box_center)
                server.scene.add_point_cloud(
                    name=f'obj_{obj_id} (camera_aligned)',
                    points=point_array,
                    point_size=0.05,
                    colors=(0, 255, 0),
                    position=pos,
                    visible=False
                )

    # Visualize ScanNet GT point cloud if path provided
    if scannet_pth_path is not None:
        import os
        if os.path.exists(scannet_pth_path):
            print(f"Loading ScanNet GT point cloud from {scannet_pth_path}...")
            gt_data = torch.load(scannet_pth_path, weights_only=False)

            gt_xyz = gt_data['xyz']
            gt_rgb = gt_data['rgb']

            server.scene.add_point_cloud(
                name="gt_point_cloud",
                points=gt_xyz,
                colors=gt_rgb,
                point_size=0.01,
                visible=True
            )
            print(f"  Added GT point cloud: {gt_xyz.shape[0]} points")

            # Add AABBs if available
            if 'aabb_corner_xyz' in gt_data and 'aabb_obj_ids' in gt_data:
                aabb_corners = gt_data['aabb_corner_xyz']
                aabb_obj_ids = gt_data['aabb_obj_ids']

                aabb_colors = [
                    (255, 100, 100),  # Light red
                    (100, 255, 100),  # Light green
                    (100, 100, 255),  # Light blue
                    (255, 255, 100),  # Yellow
                    (255, 100, 255),  # Magenta
                    (100, 255, 255),  # Cyan
                ]

                # AABB edges: 12 edges connecting 8 corners
                edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                    (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
                ]

                for i, (obj_id, corners) in enumerate(zip(aabb_obj_ids, aabb_corners)):
                    color = aabb_colors[i % len(aabb_colors)]
                    edge_points = []
                    for e0, e1 in edges:
                        edge_points.append(np.array([corners[e0], corners[e1]]))
                    edge_points = np.stack(edge_points, axis=0)

                    server.scene.add_line_segments(
                        name=f"gt_aabb_{obj_id}",
                        points=edge_points,
                        colors=color,
                        line_width=2.0,
                        visible=True
                    )
                print(f"  Added {len(aabb_obj_ids)} GT AABBs")
        else:
            print(f"Warning: ScanNet GT path not found: {scannet_pth_path}")

    # === Event Handlers (skipped in fast mode) ===
    _current_attn_colors = [attn_initial_colors]  # list-of-one for mutability

    if not fast:
        def update_frame_visibility():
            """Update frame visibility based on current mode and timestep."""
            current = gui_timestep.value
            stride = gui_stride.value
            max_visible = gui_num_frames_visible.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    if gui_accumulative.value:
                        in_range = (i <= current) and (i % stride == 0)
                        if gui_playing.value:
                            in_range = in_range and (i > current - max_visible * stride)
                        frame_node.visible = in_range
                    else:
                        frame_node.visible = (i == current)
            server.flush()

        @gui_next_frame.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        @gui_timestep.on_update
        def _(_):
            update_frame_visibility()

        @gui_accumulative.on_update
        def _(_):
            update_frame_visibility()

        @gui_stride.on_update
        def _(_):
            update_frame_visibility()

        @gui_color_mode.on_update
        def _(_):
            mode_key = _color_mode_key()
            with server.atomic():
                for idx, handle in point_cloud_handles.items():
                    if mode_key in frame_color_data[idx]:
                        handle.points = frame_points_data[idx][mode_key]
                        handle.colors = frame_color_data[idx][mode_key]
            server.flush()

        @gui_num_frames_visible.on_update
        def _(_):
            update_frame_visibility()

        @gui_show_frustums.on_update
        def _(_):
            with server.atomic():
                for frustum in frustum_handles:
                    frustum.visible = gui_show_frustums.value

        # Debounce decorator for expensive callbacks
        def debounced(delay_sec=0.15):
            def decorator(fn):
                timer_holder = [None]
                def wrapper(*args, **kwargs):
                    if timer_holder[0] is not None:
                        timer_holder[0].cancel()
                    timer_holder[0] = threading.Timer(delay_sec, fn, args, kwargs)
                    timer_holder[0].start()
                return wrapper
            return decorator

        def _update_conf_threshold():
            """Recompute points/colors for all frames after confidence threshold change."""
            mode_key = _color_mode_key()
            with server.atomic():
                for idx, handle in point_cloud_handles.items():
                    fd = per_frame_data[idx]
                    conf_mask = get_conf_mask(fd)
                    pts_m, rgb_m = apply_mask(fd['pts_3d'], fd['colors_rgb'], conf_mask)
                    _, ts_m = apply_mask(fd['pts_3d'], fd['colors_timestamp'], conf_mask)
                    color_cache = {'rgb': rgb_m, 'timestamp': ts_m}
                    points_cache = {'rgb': pts_m, 'timestamp': pts_m}
                    if _current_attn_colors[0] is not None:
                        _, attn_m = apply_mask(fd['pts_3d'], _current_attn_colors[0][idx], conf_mask)
                        color_cache['attention'] = attn_m
                        points_cache['attention'] = pts_m
                    frame_color_data[idx] = color_cache
                    frame_points_data[idx] = points_cache
                    handle.points = points_cache[mode_key]
                    handle.colors = color_cache[mode_key]
            server.flush()

        @gui_point_size.on_update
        def _(_):
            with server.atomic():
                for handle in point_cloud_handles.values():
                    handle.point_size = gui_point_size.value
            server.flush()

        if gui_conf_threshold is not None:
            @gui_conf_threshold.on_update
            @debounced(0.1)
            def _(_):
                _update_conf_threshold()

        # === Attention recomputation handler ===
        if attention_data is not None and attn_assignments is not None:
            def recompute_attention_colors():
                """Recompute attention point cloud colors based on GUI state."""
                token_idx = gui_attn_token_slider.value
                if token_idx == 0:
                    # Aggregated mode
                    attn_weights = aggregate_attention_across_tokens(
                        attention_data['attention_matrix'],
                        mode=gui_attn_aggregation.value,
                    ).numpy()
                    gui_attn_token_label.value = f"[Aggregated {gui_attn_aggregation.value}]"
                else:
                    # Single token mode (slider 1 = token index 0)
                    actual_idx = token_idx - 1
                    attn_weights = attention_data['attention_matrix'][actual_idx].numpy()
                    token_text = attention_data['generated_tokens_text'][actual_idx]
                    gui_attn_token_label.value = f"[{actual_idx}] {token_text.strip() or repr(token_text)}"

                gamma = gui_attn_gamma.value
                hide = gui_attn_hide_unassigned.value
                effective_sigma = attn_sigma * gui_attn_sigma_factor.value
                per_frame_pts3d_list = [fd['pts_3d'] for fd in per_frame_data]

                colors, masks, _ = compute_attention_colors_from_assignment(
                    attn_assignments, attn_threshold, attn_weights,
                    per_frame_pts3d_list, sigma=effective_sigma, gamma=gamma,
                    hide_unassigned=hide,
                )
                viridis_colors, _, alpha_maps = compute_attention_colors_from_assignment(
                    attn_assignments, attn_threshold, attn_weights,
                    per_frame_pts3d_list, sigma=effective_sigma, gamma=gamma,
                    hide_unassigned=hide, colormap="inferno",
                )
                _current_attn_colors[0] = colors

                cur_mode = gui_color_mode.value
                is_attn_mode = (cur_mode == "Attention Heatmap")
                is_overlay_mode = (cur_mode == "Attention Heatmap (overlay)")
                blend = gui_overlay_blend.value
                with server.atomic():
                    for idx, handle in point_cloud_handles.items():
                        if 'attention' not in frame_color_data[idx]:
                            continue
                        fd = per_frame_data[idx]
                        conf_mask = get_conf_mask(fd)
                        # Combine confidence mask with hide-unassigned mask
                        if hide:
                            combined = masks[idx].copy()
                            if conf_mask is not None:
                                combined = combined & conf_mask
                            vis_pts = fd['pts_3d'][combined]
                            vis_colors = colors[idx][combined]
                            viridis_comp = viridis_colors[idx][combined]
                            rgb_comp = fd['colors_rgb'][combined]
                            alpha_comp = alpha_maps[idx][combined]
                        else:
                            vis_pts, vis_colors = apply_mask(fd['pts_3d'], colors[idx], conf_mask)
                            _, viridis_comp = apply_mask(fd['pts_3d'], viridis_colors[idx], conf_mask)
                            _, rgb_comp = apply_mask(fd['pts_3d'], fd['colors_rgb'], conf_mask)
                            _, alpha_comp = apply_mask(fd['pts_3d'], alpha_maps[idx], conf_mask)

                        if len(vis_pts) == 0:
                            vis_pts = np.zeros((1, 3), dtype=np.float32)
                            vis_colors = np.zeros((1, 3), dtype=np.uint8)
                            viridis_comp = np.zeros((1, 3), dtype=np.uint8)
                            rgb_comp = np.zeros((1, 3), dtype=np.uint8)
                            alpha_comp = np.zeros(1, dtype=np.float32)

                        per_point_blend = (0.7 + 0.3 * alpha_comp)[:, None] * blend
                        overlay_colors = (
                            (1 - per_point_blend) * rgb_comp.astype(np.float32)
                            + per_point_blend * viridis_comp.astype(np.float32)
                        ).astype(np.uint8)

                        # Update cached colors and points
                        frame_color_data[idx]['attention'] = vis_colors
                        frame_points_data[idx]['attention'] = vis_pts
                        frame_color_data[idx]['attention_overlay'] = overlay_colors
                        frame_color_data[idx]['attention_viridis'] = viridis_comp
                        frame_color_data[idx]['attention_rgb'] = rgb_comp
                        frame_color_data[idx]['attention_alpha'] = alpha_comp
                        frame_points_data[idx]['attention_overlay'] = vis_pts

                        # Only push to handle if in attention/overlay mode
                        if is_attn_mode:
                            handle.points = vis_pts
                            handle.colors = vis_colors
                        elif is_overlay_mode:
                            handle.points = vis_pts
                            handle.colors = overlay_colors
                server.flush()

            @gui_attn_token_slider.on_update
            @debounced(0.1)
            def _(_):
                recompute_attention_colors()

            @gui_attn_aggregation.on_update
            def _(_):
                if gui_attn_token_slider.value == 0:
                    recompute_attention_colors()

            @gui_attn_hide_unassigned.on_update
            @debounced(0.1)
            def _(_):
                recompute_attention_colors()

            @gui_attn_gamma.on_update
            @debounced(0.1)
            def _(_):
                recompute_attention_colors()

            @gui_attn_sigma_factor.on_update
            @debounced(0.1)
            def _(_):
                recompute_attention_colors()

            @gui_overlay_blend.on_update
            @debounced(0.05)
            def _(_):
                blend = gui_overlay_blend.value
                is_overlay_mode = (gui_color_mode.value == "Attention Heatmap (overlay)")
                with server.atomic():
                    for idx, handle in point_cloud_handles.items():
                        viridis = frame_color_data[idx].get('attention_viridis')
                        rgb = frame_color_data[idx].get('attention_rgb')
                        alpha = frame_color_data[idx].get('attention_alpha')
                        if viridis is None or rgb is None or alpha is None:
                            continue
                        per_point_blend = (0.7 + 0.3 * alpha)[:, None] * blend
                        overlay = (
                            (1 - per_point_blend) * rgb.astype(np.float32)
                            + per_point_blend * viridis.astype(np.float32)
                        ).astype(np.uint8)
                        frame_color_data[idx]['attention_overlay'] = overlay
                        if is_overlay_mode:
                            handle.colors = overlay
                server.flush()

        # Initialize visibility
        update_frame_visibility()

    print(f"\nViser server running at: http://localhost:8080")
    print("Press Ctrl+C to stop visualization and continue...")

    try:
        while True:
            if not fast and gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1.0 / (gui_framerate.value if not fast else 1.0))
    except KeyboardInterrupt:
        viser_elapsed = time.time() - viser_start_time
        print(f"\nVisualization stopped after {viser_elapsed:.2f} seconds")
        print(f"(Subtract this from total preprocessing time for actual processing time)")


def extract_pointer_memory(
    image_inputs,
    point3r_model,
    image_embeds=None,
    grid_thw=None,
    deepstack_image_embeds=None,
    device='cuda',
    no_crop=False,
    full_seq=False,
    size=512,
    verbose=True,
    lambda_decay=1.0,
    max_memory_tokens=None,
    frames_indices=None,
    save_point3r_outputs=False,
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
        annotation_result: Output from extract_box_and_coordinates_from_scan2cap containing
                          gt_box, transformed_center, and transformation matrices for viser
                          visualization (default: None)
        scannet_pth_path: Path to ScanNet .pth file containing ground truth point cloud
                         (xyz, rgb, aabb_corner_xyz, aabb_obj_ids) for visualization (default: None)
        scannet_pose_paths: List of paths to ScanNet pose .txt files containing camera poses
                           (default: None)

    Returns:
        dict: Dictionary containing:
            - 'pointer_memory_embeds': Tensor of shape (num_pointers, 2048) or (num_pointers, 2560)
                                      Qwen image embeddings aligned with memory (used for LLM input)
            - 'pointer_positions': Tensor of shape (num_pointers, 3)
                                  3D positions (x, y, z) for each pointer in world coordinates
            - 'memory_feat': (Optional) Tensor of shape (num_pointers, 768)
                            Point3R's internal decoder features
                            Only present if the model returns this field
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
    if isinstance(size, tuple):
        target_size = size
    elif size == 512:
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
        deepstack_image_embeds=deepstack_image_embeds,
        verbose=verbose,
        lambda_decay=lambda_decay,
        max_memory_tokens=max_memory_tokens,
    )
        
    # Extract memory_aligned_image_embeds from Point3R outputs
    if 'memory_aligned_image_embeds' in outputs and outputs['memory_aligned_image_embeds'] is not None:
        memory_aligned_image_embeds = outputs['memory_aligned_image_embeds']
        if isinstance(memory_aligned_image_embeds, list):
            memory_aligned_image_embeds = memory_aligned_image_embeds[-1]
        if memory_aligned_image_embeds.dim() == 3:
            memory_aligned_image_embeds = memory_aligned_image_embeds[0]
        pointer_memory_embeds = memory_aligned_image_embeds
        if verbose:
            print(f"Extracted memory_aligned_image_embeds: {memory_aligned_image_embeds.shape}")
    else:
        raise ValueError("memory_aligned_image_embeds not found in outputs")

    # Extract memory_feat from Point3R outputs
    memory_feat = None
    if 'memory_feat' in outputs and outputs['memory_feat'] is not None:
        memory_feat = outputs['memory_feat']
        if isinstance(memory_feat, list):
            memory_feat = memory_feat[-1]
        if memory_feat.dim() == 3:
            memory_feat = memory_feat[0]
        if verbose:
            print(f"Extracted memory_feat: {memory_feat.shape}")

    # Extract deepstack_memory_aligned_embeds from Point3R outputs
    deepstack_memory_aligned_embeds = None
    if 'deepstack_memory_aligned_embeds' in outputs and outputs['deepstack_memory_aligned_embeds'] is not None:
        deepstack_memory_aligned_embeds = outputs['deepstack_memory_aligned_embeds']
        # deepstack is a list of per-layer embeddings, each is a list (per batch) or tensor
        processed_deepstack = []
        for layer_embeds in deepstack_memory_aligned_embeds:
            if isinstance(layer_embeds, list):
                layer_embeds = layer_embeds[-1]  # Take last batch element
            if layer_embeds.dim() == 3:
                layer_embeds = layer_embeds[0]  # Remove batch dimension
            processed_deepstack.append(layer_embeds)
        deepstack_memory_aligned_embeds = processed_deepstack
        if verbose:
            print(f"Extracted deepstack_memory_aligned_embeds: {len(deepstack_memory_aligned_embeds)} layers")
            for i, layer in enumerate(deepstack_memory_aligned_embeds):
                print(f"  - Layer {i}: {layer.shape}")

    # Extract memory_aligned_timestamps from Point3R outputs
    pointer_timestamps = None
    if 'memory_aligned_timestamps' in outputs and outputs['memory_aligned_timestamps'] is not None:
        memory_aligned_timestamps = outputs['memory_aligned_timestamps']
        if isinstance(memory_aligned_timestamps, list):
            memory_aligned_timestamps = memory_aligned_timestamps[-1]  # Take last batch element
        if memory_aligned_timestamps.dim() == 2:
            memory_aligned_timestamps = memory_aligned_timestamps[0]  # Remove batch dimension
        pointer_timestamps = memory_aligned_timestamps.cpu()
        if verbose:
            print(f"Extracted memory_aligned_timestamps: {pointer_timestamps.shape[0]} tokens")
            print(f"  - Timestamp range: [{pointer_timestamps.min().item()}, {pointer_timestamps.max().item()}]")
            print(f"  - Unique timestamps: {pointer_timestamps.unique().tolist()}")

    # Extract pos_decode_memory from Point3R outputs
    if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
        pos_decode_memory = outputs['pos_decode_memory']
        if isinstance(pos_decode_memory, list):
            pos_decode_memory = pos_decode_memory[-1]
        if pos_decode_memory.dim() == 3:
            pos_decode_memory = pos_decode_memory[0]

        pointer_positions = pos_decode_memory.cpu()
        if verbose:
            print(f"Extracted pos_decode_memory: {pointer_positions.shape[0]} points")
            print(f"  - xyz ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                  f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                  f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")

    # Sort all pointer data by ascending timestamp
    if pointer_timestamps is not None:
        sort_indices = torch.argsort(pointer_timestamps)
        pointer_timestamps = pointer_timestamps[sort_indices]
        pointer_memory_embeds = pointer_memory_embeds[sort_indices]
        pointer_positions = pointer_positions[sort_indices]
        if memory_feat is not None:
            memory_feat = memory_feat[sort_indices]
        if deepstack_memory_aligned_embeds is not None:
            deepstack_memory_aligned_embeds = [layer[sort_indices] for layer in deepstack_memory_aligned_embeds]
        if verbose:
            print(f"Sorted pointer data by timestamp (ascending)")

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
        if memory_feat is not None:
            print(f"  - Memory feat shape: {memory_feat.shape}")
        if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
            print(f"  - Final position ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                  f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                  f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")

    result = {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions': pointer_positions,
    }

    if save_point3r_outputs:
        result['_point3r_outputs'] = outputs

    # Add memory_feat if available
    if memory_feat is not None:
        result['memory_feat'] = memory_feat

    # Add camera poses if available
    if camera_poses is not None:
        result['camera_poses'] = camera_poses

    # Add deepstack_image_embeds if available
    if deepstack_memory_aligned_embeds is not None:
        result['deepstack_image_embeds'] = deepstack_memory_aligned_embeds

    # Add pointer_timestamps if available
    if pointer_timestamps is not None:
        result['pointer_timestamps'] = pointer_timestamps

    # Add frames_indices if available
    if frames_indices is not None:
        result['frames_indices'] = frames_indices

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
