#!/bin/bash
# Summarize token counts and max frame index for pointer memory .pt files in a folder.
# Usage: bash scripts/demo/pointer_token_stats.sh <pointer_dir> [max_files]
# Uses `python` from the current environment (override with PYTHON=/path/to/python).

DIR=${1:?"Usage: $0 <pointer_dir> [max_files]"}
LIMIT=${2:-0}
PYTHON=${PYTHON:-python}

"$PYTHON" - "$DIR" "$LIMIT" <<'EOF'
import os, sys, torch

src, limit = sys.argv[1], int(sys.argv[2])
files = sorted(f for f in os.listdir(src) if f.endswith(".pt"))
if limit > 0:
    files = files[:limit]

tokens, fmax, no_frames, failed = [], [], 0, []
for i, f in enumerate(files):
    try:
        x = torch.load(os.path.join(src, f), map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        failed.append((f, str(e)))
        continue
    tokens.append(int(x["pointer_memory_embeds"].shape[0]))
    if x.get("frames_indices") is not None and len(x["frames_indices"]) > 0:
        fmax.append(int(max(x["frames_indices"])))
    else:
        no_frames += 1
    if (i + 1) % 100 == 0:
        print(f"  ... {i + 1}/{len(files)}", file=sys.stderr, flush=True)

print(f"Directory : {src}")
print(f"Files     : {len(tokens)} read / {len(files)} found" + (f" ({len(failed)} failed)" if failed else ""))
if tokens:
    print(f"Tokens    : mean={sum(tokens) / len(tokens):.1f}  min={min(tokens)}  max={max(tokens)}")
if fmax:
    print(f"Max frame index (per scene): mean={sum(fmax) / len(fmax):.1f}  min={min(fmax)}  max={max(fmax)}")
if no_frames:
    print(f"Files without frames_indices: {no_frames}")
for f, e in failed[:10]:
    print(f"  FAILED {f}: {e}")
EOF
