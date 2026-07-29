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

1. Unparseable numeric predictions score worst-case (100 m) instead of being dropped
   from the RMSE. Upstream's `if pred:` also drops legitimate `0` predictions.
   Strictly harsher than `Ego3D-Bench/utils/eval.py`.
2. No resume logic. Upstream's `idx < processed` check is off by one.
3. For `Ego_Centric_Relative_Distance`, `Ego_Centric_Motion_Reasoning` and
   `Object_Centric_Motion_Reasoning`, the ground truth is the option LETTER
   (`A`/`B`) but the prompt suffix asks for "final answer (yes or no)" and the
   options are text-valued (`['A. yes', 'B. no']`). A model that correctly answers
   with the option text (e.g. `<answer>yes</answer>`) is credited by resolving the
   prediction against the doc's own `options` list generically — not a yes/no
   special case, so it also covers any other text-valued option. This is a
   correctness fix, not a harshness change: upstream's own scorer
   (`Ego3D-Bench/utils/eval.py`) has this same yes/no-versus-letter gap and would
   also score a correct `yes`/`no` answer as wrong for these three categories.
4. Numeric predictions are clipped to `[0, 100]`; upstream only upper-clips at 100
   (no lower clip). Undocumented until now; strictly harsher for negative
   predictions (rare, but possible from a malformed extraction).
5. The benchmark's own option strings are inconsistently formatted — some have a
   space after the letter marker (`"A. yes"`), some don't (`"A.36 meters"`,
   `"A.ego car"`) — and a model in pointer mode tends to echo the option text
   verbatim, producing predictions like `"A.1 meter"` or `"A.ego car"` rather than
   a bare letter or a cleanly separated one. We recognize a leading option letter
   followed by an optional `.`, `)`, or `:` separator even when glued directly to
   the option text, guarded by the doc's own option count (`A..<len(options)>`)
   so that text answers such as `"no"`/`"yes"` are never misread as letters
   `N`/`Y` — those still resolve through the deviation-3 option-TEXT matching.
   Upstream's scorer (`Ego3D-Bench/utils/eval.py`) takes only the first
   whitespace-delimited token and strips a trailing period, so it has this same
   glued-letter gap; our port is therefore more permissive in what it credits.

## Reading the numbers

Aggregation prints each category's score next to its trivial floor. The RMSE floors
matter most: always answering "15" scores 8.4 / 10.2, which beats every open-source
baseline in the paper (Qwen2.5-3B: 30.5 / 33.7) and matches its Ego3D-VLM results
(6.3-6.8 / 7.8-8.4). RMSE here largely measures whether a model emits numbers in the
plausible outdoor range. Read any RMSE gain against those floors.
