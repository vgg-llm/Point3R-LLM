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
    to the first token (attention sink, e.g. <|im_start|> in Qwen3-VL).

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
    # Zero out first-token attention sink (e.g. <|im_start|> in Qwen3-VL).
    # When keep_indices is used, non-pointer positions are already zeroed,
    # making this a no-op.
    if keep_indices is None:
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


def precompute_dense_point_assignment(per_frame_pts3d, pointer_positions, pointer_timestamps,
                                       threshold_factor=1.0, k=8):
    """Pre-compute the K-nearest-pointer assignment for each dense point.

    Builds a single global KD-tree from all pointer positions and finds the K
    nearest pointers for each dense 3D point (across all frames). This is done
    once and cached so that changing attention weights only requires an O(N)
    lookup with Gaussian-weighted blending.

    Args:
        per_frame_pts3d: list of (N_i, 3) numpy arrays (dense points per frame).
        pointer_positions: (num_pointers, 3) numpy array.
        pointer_timestamps: (num_pointers,) numpy array of frame indices (unused,
            kept for API compatibility).
        threshold_factor: multiplier on the auto-computed inclusion threshold.
        k: number of nearest pointers per dense point for Gaussian blending.

    Returns:
        assignments: list of dicts per frame, each with:
            'global_indices': (N_i, k_actual) int array — indices into pointer_positions
            'distances': (N_i, k_actual) float array — distances to nearest pointers
        threshold: float — the computed inclusion threshold.
        sigma: float — Gaussian kernel width (median NN distance among pointers).
    """
    from scipy.spatial import cKDTree

    # Compute adaptive threshold from median nearest-neighbor distance
    if len(pointer_positions) > 1:
        tree_all = cKDTree(pointer_positions)
        dists_all, _ = tree_all.query(pointer_positions, k=2)  # k=2: self + nearest
        threshold = float(np.median(dists_all[:, 1])) * threshold_factor
    else:
        threshold = float('inf')

    sigma = threshold  # Gaussian width = median pointer spacing

    # Build single global KD-tree from all pointers (no per-frame restriction)
    k_actual = min(k, len(pointer_positions))
    tree = cKDTree(pointer_positions)

    assignments = []
    for pts_3d in per_frame_pts3d:
        distances, global_indices = tree.query(pts_3d, k=k_actual)

        # When k_actual == 1, cKDTree returns (N,) — normalize to 2D
        if k_actual == 1:
            distances = distances[:, np.newaxis]
            global_indices = global_indices[:, np.newaxis]

        assignments.append({
            'global_indices': global_indices,
            'distances': distances,
        })

    return assignments, threshold, sigma


def compute_attention_colors_from_assignment(assignments, threshold, attention_weights,
                                              per_frame_pts3d, sigma=None, colormap="inferno",
                                              gamma=0.2, hide_unassigned=False):
    """Compute per-frame RGB colors using Gaussian-weighted attention blending.

    Each dense point's attention is a weighted sum of its K nearest pointers'
    attention values, with weights decaying as a 3D Gaussian: w = exp(-d²/(2σ²)).
    All points are colored — the Gaussian naturally handles distance falloff.

    Args:
        assignments: from precompute_dense_point_assignment().
        threshold: unused, kept for API compatibility.
        attention_weights: (num_pointers,) numpy array.
        per_frame_pts3d: list of (N_i, 3) numpy arrays (only used for shape).
        sigma: Gaussian kernel width. If None, uses threshold as fallback.
        colormap: matplotlib colormap name.
        gamma: gamma correction exponent.
        hide_unassigned: unused, kept for API compatibility.

    Returns:
        per_frame_colors: list of (N_i, 3) uint8 arrays.
        per_frame_masks: list of (N_i,) bool arrays (all True).
        per_frame_alpha: list of (N_i,) float32 arrays (normalized attention, 0-1).
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)

    if sigma is None:
        sigma = threshold
    two_sigma_sq = 2.0 * sigma * sigma

    # First pass: compute Gaussian-weighted attention per point, collect for normalization
    per_frame_weighted_attn = []
    all_valid_values = []

    for assignment in assignments:
        if assignment is None:
            per_frame_weighted_attn.append(None)
            continue

        dists = assignment['distances']       # (N, K)
        gidx = assignment['global_indices']   # (N, K)

        ptr_attn = attention_weights[gidx]    # (N, K)

        # Gaussian weights: w_i = exp(-d_i² / (2σ²))
        gauss_weights = np.exp(-dists**2 / two_sigma_sq)  # (N, K)
        # Zero out infinite distances (when fewer pointers than K)
        gauss_weights[~np.isfinite(dists)] = 0.0

        numerator = (gauss_weights * ptr_attn).sum(axis=1)       # (N,)
        denominator = np.maximum(gauss_weights.sum(axis=1), 1e-12)  # (N,)
        weighted_attn = numerator / denominator                   # (N,)

        all_valid_values.append(weighted_attn)
        per_frame_weighted_attn.append(weighted_attn)

    # Global normalization range
    if len(all_valid_values) > 0:
        all_valid_concat = np.concatenate(all_valid_values)
        global_vmin, global_vmax = float(all_valid_concat.min()), float(all_valid_concat.max())
    else:
        global_vmin, global_vmax = 0.0, 1.0

    # Second pass: apply colormap
    per_frame_colors = []
    per_frame_masks = []
    per_frame_alpha = []

    for frame_idx, cached in enumerate(per_frame_weighted_attn):
        num_pts = per_frame_pts3d[frame_idx].shape[0]

        if cached is None:
            per_frame_colors.append(np.full((num_pts, 3), 40, dtype=np.uint8))
            per_frame_masks.append(np.zeros(num_pts, dtype=bool))
            per_frame_alpha.append(np.zeros(num_pts, dtype=np.float32))
            continue

        weighted_attn = cached
        if global_vmax > global_vmin:
            normalized = (weighted_attn - global_vmin) / (global_vmax - global_vmin)
        else:
            normalized = np.ones_like(weighted_attn) * 0.5
        normalized = np.power(np.clip(normalized, 0, 1), gamma)
        rgba = cmap(normalized)
        colors = (rgba[:, :3] * 255).astype(np.uint8)

        per_frame_colors.append(colors)
        per_frame_masks.append(np.ones(num_pts, dtype=bool))
        per_frame_alpha.append(normalized.astype(np.float32))

    return per_frame_colors, per_frame_masks, per_frame_alpha
