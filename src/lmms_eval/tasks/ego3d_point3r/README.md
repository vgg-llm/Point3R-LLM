# Ego3D-Bench (Point3R)

8,675 ego-centric multi-view outdoor spatial QA pairs over 262 scenes from nuScenes,
Waymo and Argoverse. Test split only — the benchmark ships no training data.

## Tasks

| Task | Visuals | Protocol | Use |
|---|---|---|---|
| `ego3d_baseline` | 5-7 images | official `<think>/<answer>` | images-only reference run |
| `ego3d_point3r` | pointer tokens | short answers | main Point3R run |
| `ego3d_point3r_think` | pointer tokens | official `<think>/<answer>` | protocol-matched comparison |

The three tasks share `ego3d_default_yaml` via `include:`; each task yaml overrides
only the visual substrate, the generation length and the protocol.

### Prompts are NOT identical across modes

Both prompts carry the same question body and options, and a manifest header naming
each camera view. They differ in two ways, both forced by the visual substrate:

| | pointer modes | `ego3d_baseline` |
|---|---|---|
| pointer-pad line | `<|pointer_pad|>` present | absent |
| manifest frame labels | `<frame-1>..<frame-N>` | `Frame-0..Frame-N-1` |

The frame labels must match how the wrapper actually labels the visual tokens, and the
two wrappers disagree: `src/qwen_vl/data/pointer_data.py` (`add_frame_id=true`) groups
pointer tokens as `<frame-1>`, `<frame-2>`, ... (1-indexed, bracketed, lowercase),
while `src/lmms_eval/models/point3r_llm_v2.py` (`add_frame_index=true`) prefixes each
image with `Frame-0: `, `Frame-1: `, ... (0-indexed). So a baseline/pointer comparison
varies the visual substrate **plus the frame-naming that substrate dictates** — it is
not a single-line delta. Run the baseline with `add_frame_index=true` and pointer mode
with `add_frame_id=true`, as `scripts/run/ego3d_eval.sh` does.

### The default baseline/pointer comparison also varies the WEIGHTS

`scripts/run/ego3d_eval.sh` defaults to *different checkpoints* per mode:

| Mode | Default checkpoint |
|---|---|
| `baseline` | stock `Qwen/Qwen3-VL-4B-Instruct` |
| `pointer`, `pointer_think` | `./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5` (finetuned) |

Any headline gap between those two runs therefore mixes a weights difference with the
substrate difference and is **not** a measurement of the pointer substrate. Every mode
honors `MODEL_PATH`, so hold the weights fixed for a controlled comparison:

    MODEL_PATH=./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5 \
        bash scripts/run/ego3d_eval.sh baseline

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
   The leading letter must END the token or be followed by a non-letter, so free
   text (`"approximately 12 meters"`, `"cannot determine"`) is never credited as
   option `A`/`C`. Wrapping punctuation (`.,;:!*)"'`) is stripped from both ends of
   the token and of the full span, so `"**B**"` and
   `"yes, it is moving toward the ego car"` resolve to `B` and `A`. As a last
   resort a bracketed/starred or span-final option letter is accepted
   (`"the answer is (B)."`), guarded by the doc's option count.
6. A numeric answer is only read from a properly CLOSED `<answer></answer>` span
   under the think protocol (`utils.ego3d_process_results_think`, wired by
   `ego3d_baseline.yaml` and `ego3d_point3r_think.yaml`); the short protocol, which
   asks for "a single word or phrase" and caps generation at 16 tokens, is also read
   untagged. A think response truncated at the 1024-token cap therefore scores the
   worst case (deviation 1) instead of having a number mined out of its reasoning.
   The number regex additionally refuses a digit glued to a word character, `.` or
   `-`, so a `Frame-1` / `<frame-1>` reference can never be read as the number `-1`.
   Strictly harsher than upstream, and it invalidates any RMSE produced before this
   fix: the pre-fix baseline RMSEs were shaped by numbers mined from reasoning.

## Reading the numbers

Aggregation prints each category's score next to its trivial floor. The RMSE floors
matter most: always answering "15" scores 8.4 / 10.2, which beats every open-source
baseline in the paper (Qwen2.5-3B: 30.5 / 33.7) and matches its Ego3D-VLM results
(6.3-6.8 / 7.8-8.4). RMSE here largely measures whether a model emits numbers in the
plausible outdoor range. Read any RMSE gain against those floors.
