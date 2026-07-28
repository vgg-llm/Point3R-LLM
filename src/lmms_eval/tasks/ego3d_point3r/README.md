# Ego3D-Bench (Point3R)

8,675 ego-centric multi-view outdoor spatial QA pairs over 262 scenes from nuScenes,
Waymo and Argoverse. Test split only — the benchmark ships no training data.

## Tasks

| Task | Visuals | Protocol | Use |
|---|---|---|---|
| `ego3d_baseline` | 5-7 images | official `<think>/<answer>` | images-only reference run |
| `ego3d_point3r` | pointer tokens | short answers | main Point3R run |
| `ego3d_point3r_think` | pointer tokens | official `<think>/<answer>` | protocol-matched comparison |

Prompts are identical across modes except for the `<|pointer_pad|>` line, so a
baseline/pointer comparison varies the visual substrate and nothing else. Views are
named `Frame-0..N` in a manifest header; run the baseline with `add_frame_index=true`
and pointer mode with `add_frame_id=true` so the visual tokens carry matching labels.

## Setup

    python scripts/preprocess/convert_ego3d.py
    CUDA_VISIBLE_DEVICES=0 python scripts/demo/preprocess_ego3d_simple.py --gpu-id 0 --total-gpus 1

## Deviations from upstream scoring

Both are strictly harsher than `Ego3D-Bench/utils/eval.py`:

1. Unparseable numeric predictions score worst-case (100 m) instead of being dropped
   from the RMSE. Upstream's `if pred:` also drops legitimate `0` predictions.
2. No resume logic. Upstream's `idx < processed` check is off by one.

## Reading the numbers

Aggregation prints each category's score next to its trivial floor. The RMSE floors
matter most: always answering "15" scores 8.4 / 10.2, which beats every open-source
baseline in the paper (Qwen2.5-3B: 30.5 / 33.7) and matches its Ego3D-VLM results
(6.3-6.8 / 7.8-8.4). RMSE here largely measures whether a model emits numbers in the
plausible outdoor range. Read any RMSE gain against those floors.
