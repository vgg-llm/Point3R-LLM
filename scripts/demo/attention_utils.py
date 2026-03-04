"""
Utility functions for extracting and visualizing attention maps
from Qwen3VL + Point3R models.

Adapted from VLM-Visualizer (sample_codes/VLM-Visualizer/utils.py).
"""

import torch
import numpy as np
import cv2


def identify_pointer_indices(input_ids, pointer_token_id):
    """Find positions of pointer tokens in the input sequence.

    Args:
        input_ids: (seq_len,) tensor of token IDs
        pointer_token_id: the ID of the pointer token

    Returns:
        pointer_indices: (num_pointers,) tensor of positions
    """
    return torch.where(input_ids == pointer_token_id)[0]


def aggregate_qwen3vl_attention(step_attentions, layer_indices=None, keep_indices=None):
    """Average attention across layers and heads for one generation step.

    Follows VLM-Visualizer's null-attention practice: zeroes the attention
    to the first token (BOS) since it acts as an attention sink.

    Args:
        step_attentions: tuple of num_layers tensors, each (1, num_heads, q_len, kv_len)
        layer_indices: Optional list/range of layer indices to average over.
                       Supports negative indices. If None, uses all layers.
        keep_indices: Optional tensor of token positions to keep.
                      All other positions are zeroed per-layer before averaging,
                      so each layer's distribution over kept tokens is weighted
                      equally. If None, keeps all positions.

    Returns:
        avg_attn: (kv_len,) averaged attention vector for the last query token
    """
    num_layers = len(step_attentions)
    if layer_indices is None:
        layers_to_use = range(num_layers)
    else:
        layers_to_use = [i % num_layers for i in layer_indices]

    per_layer = []
    for layer_idx in layers_to_use:
        layer_attn = step_attentions[layer_idx]
        # (1, num_heads, q_len, kv_len) -> (num_heads, q_len, kv_len)
        layer_attn = layer_attn.squeeze(0)
        # Average across heads, take last query position
        avg_over_heads = layer_attn.mean(dim=0)[-1]  # (kv_len,)
        attn = avg_over_heads.cpu()

        # Zero non-kept positions per-layer and renormalize
        if keep_indices is not None:
            mask = torch.zeros_like(attn)
            mask[keep_indices] = 1.0
            attn = attn * mask
            total = attn.sum()
            if total > 0:
                attn = attn / total

        per_layer.append(attn)

    avg = torch.stack(per_layer).mean(dim=0)  # (kv_len,)
    # Zero out BOS attention (null attention pattern)
    avg[0] = 0.0
    # Re-normalize
    total = avg.sum()
    if total > 0:
        avg = avg / total
    return avg


def extract_pointer_attention(outputs, pointer_indices, input_len, layer_indices=None):
    """Extract attention from each generated token to pointer tokens.

    Args:
        outputs: GenerateDecoderOnlyOutput with attentions
        pointer_indices: (num_pointers,) tensor of pointer positions in input
        input_len: length of the input sequence (before generation)
        layer_indices: Optional list of layer indices for aggregation.

    Returns:
        attention_matrix: (num_generated_tokens, num_pointer_tokens) tensor
    """
    num_generated = len(outputs.attentions)
    num_pointers = len(pointer_indices)
    attention_matrix = torch.zeros(num_generated, num_pointers)

    for gen_step, step_attns in enumerate(outputs.attentions):
        avg_attn = aggregate_qwen3vl_attention(
            step_attns, layer_indices=layer_indices, keep_indices=pointer_indices.cpu()
        )
        # Extract attention to pointer token positions only
        pointer_attn = avg_attn[pointer_indices.cpu()]
        # Re-normalize to show relative attention distribution over pointers
        ptr_sum = pointer_attn.sum()
        if ptr_sum > 0:
            pointer_attn = pointer_attn / ptr_sum
        attention_matrix[gen_step] = pointer_attn

    return attention_matrix


def build_pointer_grid(pointer_timestamps, attention_weights):
    """Arrange pointer attention weights in a frames x tokens_per_frame grid.

    Args:
        pointer_timestamps: (num_pointers,) tensor of frame indices
        attention_weights: (num_pointers,) attention weights for one generated token

    Returns:
        grid: 2D numpy array (num_frames, max_tokens_per_frame)
        frame_labels: list of frame index labels
    """
    unique_frames = pointer_timestamps.unique(sorted=True)
    tokens_per_frame = [(pointer_timestamps == f).sum().item() for f in unique_frames]
    max_tokens = max(tokens_per_frame)

    grid = np.full((len(unique_frames), max_tokens), np.nan)
    for i, frame in enumerate(unique_frames):
        mask = pointer_timestamps == frame
        frame_weights = attention_weights[mask]
        grid[i, :len(frame_weights)] = frame_weights.numpy()

    return grid, [f.item() for f in unique_frames]


def show_mask_on_image(img, mask):
    """Overlay attention heatmap on an image.

    Args:
        img: numpy array (H, W, 3) in uint8
        mask: numpy array (H, W) in [0, 1]

    Returns:
        cam: overlayed image (H, W, 3) uint8
        heatmap: raw heatmap (H, W, 3) uint8
    """
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def heterogenous_stack(vecs):
    """Pad vectors with zeros then stack (from VLM-Visualizer)."""
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def aggregate_attention_across_tokens(attention_matrix, mode="mean", token_indices=None):
    """Aggregate attention across generated tokens to get per-pointer weights.

    Args:
        attention_matrix: (num_generated_tokens, num_pointer_tokens) tensor
        mode: "mean" averages, "max" takes max, "sum" sums then normalizes.
        token_indices: Optional int or list[int] of generated token indices.
                       If None, uses all generated tokens.

    Returns:
        attention_weights: (num_pointer_tokens,) tensor
    """
    if token_indices is not None:
        if isinstance(token_indices, int):
            return attention_matrix[token_indices]
        attention_matrix = attention_matrix[token_indices]

    if mode == "mean":
        return attention_matrix.mean(dim=0)
    elif mode == "max":
        return attention_matrix.max(dim=0).values
    elif mode == "sum":
        result = attention_matrix.sum(dim=0)
        total = result.sum()
        if total > 0:
            result = result / total
        return result
    else:
        raise ValueError(f"Unknown mode: {mode}")


def precompute_dense_point_assignment(per_frame_pts3d, pointer_positions, pointer_timestamps, threshold_factor=1.0):
    """Pre-compute the nearest-pointer assignment for each dense point.

    Builds a KD-tree per frame and finds the nearest pointer token for each
    dense 3D point, restricted to same-timestamp pointers. This is done once
    and cached so that changing attention weights only requires an O(N) lookup.

    Args:
        per_frame_pts3d: list of (N_i, 3) numpy arrays (dense points per frame).
        pointer_positions: (num_pointers, 3) numpy array.
        pointer_timestamps: (num_pointers,) numpy array of frame indices.
        threshold_factor: multiplier on the auto-computed inclusion threshold.

    Returns:
        assignments: list of dicts per frame, each with:
            'global_indices': (N_i,) int array — index into pointer_positions for nearest pointer
            'distances': (N_i,) float array — distance to nearest pointer
            None if no pointers exist for that frame.
        threshold: float — the computed inclusion threshold.
    """
    from scipy.spatial import cKDTree

    # Compute adaptive threshold from median nearest-neighbor distance
    if len(pointer_positions) > 1:
        tree_all = cKDTree(pointer_positions)
        dists_all, _ = tree_all.query(pointer_positions, k=2)  # k=2: self + nearest
        threshold = float(np.median(dists_all[:, 1])) * threshold_factor
    else:
        threshold = float('inf')

    assignments = []
    num_frames = len(per_frame_pts3d)

    for frame_idx in range(num_frames):
        pts_3d = per_frame_pts3d[frame_idx]
        frame_mask = pointer_timestamps == frame_idx

        if frame_mask.sum() == 0:
            assignments.append(None)
            continue

        frame_ptr_pos = pointer_positions[frame_mask]
        global_indices_map = np.where(frame_mask)[0]  # map local→global pointer index

        tree = cKDTree(frame_ptr_pos)
        distances, local_indices = tree.query(pts_3d, k=1)
        global_indices = global_indices_map[local_indices]

        assignments.append({
            'global_indices': global_indices,
            'distances': distances,
        })

    return assignments, threshold


def compute_attention_colors_from_assignment(assignments, threshold, attention_weights,
                                              per_frame_pts3d, colormap="inferno",
                                              gamma=0.2, hide_unassigned=False):
    """Compute per-frame RGB colors from pre-computed assignment and attention weights.

    Args:
        assignments: from precompute_dense_point_assignment().
        threshold: inclusion threshold.
        attention_weights: (num_pointers,) numpy array.
        per_frame_pts3d: list of (N_i, 3) numpy arrays (only used for shape).
        colormap: matplotlib colormap name.
        gamma: gamma correction exponent.
        hide_unassigned: if True, return masks for filtering out unassigned points.

    Returns:
        per_frame_colors: list of (N_i, 3) uint8 arrays.
        per_frame_masks: list of (N_i,) bool arrays (True = assigned).
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)

    # Compute global min/max of assigned attention for consistent normalization
    all_valid = []
    for frame_idx, assignment in enumerate(assignments):
        if assignment is None:
            continue
        within = assignment['distances'] <= threshold
        if within.any():
            all_valid.append(attention_weights[assignment['global_indices'][within]])
    if len(all_valid) > 0:
        all_valid = np.concatenate(all_valid)
        global_vmin, global_vmax = float(all_valid.min()), float(all_valid.max())
    else:
        global_vmin, global_vmax = 0.0, 1.0

    per_frame_colors = []
    per_frame_masks = []

    for frame_idx, assignment in enumerate(assignments):
        num_pts = per_frame_pts3d[frame_idx].shape[0]

        if assignment is None:
            per_frame_colors.append(np.full((num_pts, 3), 40, dtype=np.uint8))
            per_frame_masks.append(np.zeros(num_pts, dtype=bool))
            continue

        within = assignment['distances'] <= threshold
        mask = within
        colors = np.full((num_pts, 3), 40, dtype=np.uint8)  # dark gray default

        if mask.any():
            valid_attn = attention_weights[assignment['global_indices'][mask]]
            if global_vmax > global_vmin:
                normalized = (valid_attn - global_vmin) / (global_vmax - global_vmin)
            else:
                normalized = np.ones_like(valid_attn) * 0.5
            normalized = np.power(np.clip(normalized, 0, 1), gamma)
            rgba = cmap(normalized)
            colors[mask] = (rgba[:, :3] * 255).astype(np.uint8)

        per_frame_colors.append(colors)
        per_frame_masks.append(mask)

    return per_frame_colors, per_frame_masks
