import sys
sys.path.insert(0, 'src')

import torch
from PIL import Image
from demo_point3r import load_models, run_vla_loop

# ── 1. Pick a model and load ──────────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen3.5-4B"          # or any checkpoint in ./outputs/

model, processor, min_pixels, max_pixels, point3r_model = load_models(
    model_path=MODEL_PATH,
    load_point3r=True,
    use_merge=True,
)

# ── 2. Build a tiny frame sequence (use any images you have) ──────────────────
from pathlib import Path
image_dir = Path("data/media/scannet/posed_images/scene0000_00")     # adjust path
jpg_paths = sorted(image_dir.glob("*.jpg"))[:32] if image_dir.is_dir() else []

if jpg_paths:
    # Option A — real images from a scene folder
    frames = [Image.open(p).convert("RGB") for p in jpg_paths]
    print(f"Loaded {len(frames)} real frames from {image_dir}")
else:
    # Option B — synthetic noise frames (no real data needed)
    print(f"[Warning] {image_dir} not found; using synthetic frames")
    frames = [Image.fromarray(torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy())
              for _ in range(5)]

# ── 3. Define a per-step query ────────────────────────────────────────────────
QUERIES = [
    "Describe the room layout.",
]
query_fn = lambda idx: QUERIES[idx % len(QUERIES)]

# ── 4. Run the loop ───────────────────────────────────────────────────────────
for frame_idx, response in run_vla_loop(
    model, processor, min_pixels, max_pixels, point3r_model,
    frame_source=frames,
    query_fn=query_fn,
    max_memory_tokens=256,
):
    print(f"\n=== Frame {frame_idx} ===")
    print(f"Q: {query_fn(frame_idx)}")
    print(f"A: {response}")