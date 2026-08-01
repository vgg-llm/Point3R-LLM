#!/usr/bin/env python
"""Convert the Ego3D-Bench HuggingFace test split into our lmms_eval task format.

Outputs:
  data/evaluation/ego3d_point3r/test.json    one doc per QA pair (8675)
  data/evaluation/ego3d_point3r/scenes.json  one entry per unique view-set (262)
  data/media/ego3d/scenes/<scene_id>/NN_<View_Key>.jpg   ordered symlinks

Usage:
  python scripts/preprocess/convert_ego3d.py
  python scripts/preprocess/convert_ego3d.py --limit 20 --out /tmp/ego3d_smoke
"""

import argparse
import json
import re
from pathlib import Path

POINTER_PAD = "<|pointer_pad|>"

# Ordered `images` dict keys, verbatim from Ego3D-Bench/models/qwen2.5_vl.py.
VIEW_ORDER = {
    "nuscenes": ["Front_Left", "Front", "Front_Right", "Back_Right", "Back", "Back_Left"],
    "waymo": ["Front", "Front_Left", "Side_Left", "Front_Right", "Side_Right"],
    "argoverse": [
        "Front_Left", "Front", "Front_Right", "Side_Right",
        "Back_Right", "Back_Left", "Side_Left",
    ],
}

# Label wording as it appears in the questions. Positionally aligned with VIEW_ORDER.
# Argoverse deliberately differs from its dict keys: "Right"/"Left", not "Side Right"/"Side Left".
EXPECTED_LABELS = {
    "nuscenes": ["Front Left", "Front", "Front Right", "Back Right", "Back", "Back Left"],
    "waymo": ["Front", "Front Left", "Side Left", "Front Right", "Side Right"],
    "argoverse": [
        "Front Left", "Front", "Front Right", "Right",
        "Back Right", "Back Left", "Left",
    ],
}

# Intentionally duplicated in src/lmms_eval/tasks/ego3d_point3r/utils.py: this
# converter must stay importable without src/lmms_eval on the path (it runs before
# any eval, in a plain `datasets`+PIL environment), so it cannot import from there.
EXACT_NUMBER_CATEGORIES = {
    "Ego_Centric_Absolute_Distance",
    "Object_Centric_Absolute_Distance",
}

_LABEL_RE = re.compile(r"([A-Za-z][A-Za-z ]*?)\s+view\s*:\s*$")


def parse_view_labels(question):
    """Return view labels in order, read from the text preceding each <image>."""
    parts = question.split("<image>")
    labels = []
    for part in parts[:-1]:
        tail = part.rstrip().splitlines()[-1].strip()
        match = _LABEL_RE.search(tail)
        if match is None:
            raise ValueError(f"could not parse view label from segment: {tail!r}")
        labels.append(match.group(1).strip())
    return labels


def strip_question(question):
    """Return the question text after the final <image> marker."""
    return question.split("<image>")[-1].strip()


# The manifest must name the frames exactly as the model wrapper labels them, and the
# two visual substrates label them differently:
#   IMAGES  path: src/lmms_eval/models/point3r_llm_v2.py (add_frame_index=true) emits
#                 "Frame-0: ", "Frame-1: ", ... -> 0-indexed, capital F, no brackets.
#   POINTER path: src/qwen_vl/data/pointer_data.py (add_frame_id=true) emits
#                 "<frame-1>", "<frame-2>", ... -> 1-indexed, lowercase f, brackets.
# Both are the models' contract; the prompt adapts to the model, not the reverse.
BASELINE_FRAME_LABEL = "Frame-{index}"
BASELINE_INDEX_BASE = 0
POINTER_FRAME_LABEL = "<frame-{index}>"
POINTER_INDEX_BASE = 1


def build_manifest(labels, index_base=BASELINE_INDEX_BASE,
                   label_template=BASELINE_FRAME_LABEL):
    """Return the manifest header naming each view, one line per view.

    `index_base` and `label_template` select the frame-naming style of the visual
    substrate the prompt is for (see BASELINE_/POINTER_FRAME_LABEL above). Defaults
    to the images/baseline style.
    """
    lines = [
        f"The scene is provided as {len(labels)} camera views mounted on an ego car, "
        "in this order:"
    ]
    lines += [
        f"  {label_template.format(index=i + index_base)}: {label}"
        for i, label in enumerate(labels)
    ]
    return "\n".join(lines)


def build_prompts(sample):
    """Return (pointer_prompt, baseline_prompt).

    The two share the same question body and options, and differ in exactly the two
    ways the visual substrate forces them to: the pointer prompt carries the
    `<|pointer_pad|>` line, and its manifest names frames `<frame-1>..<frame-N>` to
    match the pointer token groups, where the baseline manifest names them
    `Frame-0..Frame-N-1` to match the per-image text labels. A baseline/pointer
    comparison therefore varies the visual substrate plus the frame-naming those
    substrates dictate -- it is not a single-line delta.
    """
    source = sample["source"]
    labels = parse_view_labels(sample["question"])
    expected = EXPECTED_LABELS.get(source)
    if expected is not None and labels != expected:
        raise ValueError(
            f"label mismatch for source {source}: parsed {labels}, expected {expected}"
        )

    body = strip_question(sample["question"])
    options = sample.get("options") or []
    tail = "\n".join([body] + list(options))

    baseline_manifest = build_manifest(labels)
    pointer_manifest = build_manifest(
        labels, index_base=POINTER_INDEX_BASE, label_template=POINTER_FRAME_LABEL
    )
    baseline_prompt = f"{baseline_manifest}\n\n{tail}"
    pointer_prompt = f"{pointer_manifest}\n{POINTER_PAD}\n\n{tail}"
    return pointer_prompt, baseline_prompt


def scene_id_of(sample):
    front = sample["images"]["Front"]
    if not front:
        raise ValueError(f"sample {sample['idx']} has no Front view")
    return Path(front).stem


POINTER_DIR = "ego3d/pointer_memory_qwen3vl"
SCENES_DIR = "ego3d/scenes"


def build_scene(sample):
    """Return {scene_id, source, image_names} with names in canonical view order."""
    # VIEW_ORDER[source], not .get(): link_scene indexes it directly, so an unknown
    # source must fail here too rather than half-succeeding with a guessed key order
    # that link_scene would then reject (or, worse, mislabel).
    source = sample["source"]
    keys = VIEW_ORDER[source]
    names = [sample["images"][key] for key in keys]
    if not all(names):
        raise ValueError(f"sample {sample['idx']} missing an image for source {source}")
    return {"scene_id": scene_id_of(sample), "source": source, "image_names": names}


def _link_names(sample):
    """Return ordered symlink filenames: 00_<View_Key>.jpg, 01_..., preserving suffix."""
    keys = VIEW_ORDER[sample["source"]]
    suffixes = [Path(sample["images"][key]).suffix for key in keys]
    return [f"{i:02d}_{key}{suffix}" for i, (key, suffix) in enumerate(zip(keys, suffixes))]


def build_doc(sample):
    pointer_prompt, baseline_prompt = build_prompts(sample)
    scene_id = scene_id_of(sample)
    category = sample["category"]
    answer = str(sample["answer"])
    return {
        "conversations": [
            {"from": "human", "value": pointer_prompt},
            {"from": "gpt", "value": answer},
        ],
        "answer": answer,
        "options": list(sample.get("options") or []),
        "baseline_prompt": baseline_prompt,
        "pointer_data": f"{POINTER_DIR}/{scene_id}.pt",
        "images": [f"{SCENES_DIR}/{scene_id}/{name}" for name in _link_names(sample)],
        "metadata": {
            "idx": int(sample["idx"]),
            "source": sample["source"],
            "category": category,
            "scene_id": scene_id,
            "question_type": (
                "exact_number" if category in EXACT_NUMBER_CATEGORIES else "multi_choice"
            ),
        },
    }


def link_scene(scene, image_root, scenes_root):
    """Create <scenes_root>/<scene_id>/NN_<View>.jpg symlinks. Returns the scene dir."""
    keys = VIEW_ORDER[scene["source"]]
    scene_dir = Path(scenes_root) / scene["scene_id"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    for i, (key, name) in enumerate(zip(keys, scene["image_names"])):
        source_path = Path(image_root) / name
        if not source_path.exists():
            raise FileNotFoundError(f"missing Ego3D-Bench image: {source_path}")
        link_path = scene_dir / f"{i:02d}_{key}{source_path.suffix}"
        if link_path.exists() and not link_path.is_symlink():
            # Already normalized (padded) in place by normalize_scene_resolution; leave
            # the real file alone so re-running the converter doesn't destroy padding.
            continue
        if link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_path.resolve())
    return scene_dir


def normalize_scene_resolution(scene_dir):
    """Pad every view in a scene dir to the scene's max width/height, black, centered.

    Point3R asserts all views of a scene share one grid, but waymo rigs mix
    1920x1280 and 1920x886 within a scene. Mirrors the semantics of
    Ego3D-Bench/utils/common.py:pad_images. Padding replaces the symlink with a real
    JPEG under the same name, so pointer extraction and the images baseline see
    identical pixels. Returns True if anything was padded.
    """
    from PIL import Image, ImageOps

    paths = sorted(p for p in Path(scene_dir).iterdir() if p.suffix.lower() == ".jpg")
    sizes = {}
    for path in paths:
        with Image.open(path) as image:
            sizes[path] = image.size
    if len(set(sizes.values())) <= 1:
        return False

    max_width = max(width for width, _ in sizes.values())
    max_height = max(height for _, height in sizes.values())

    for path, (width, height) in sizes.items():
        if (width, height) == (max_width, max_height):
            continue
        delta_w, delta_h = max_width - width, max_height - height
        padding = (
            delta_w // 2, delta_h // 2,
            delta_w - delta_w // 2, delta_h - delta_h // 2,
        )
        with Image.open(path) as image:
            padded = ImageOps.expand(image.convert("RGB"), padding, fill=0)
        # Write to a temp path then replace, so the symlink is swapped atomically for a
        # real file and an interrupted run cannot leave a half-written JPEG.
        tmp_path = path.with_suffix(".jpg.tmp")
        padded.save(tmp_path, "JPEG", quality=95)
        path.unlink()
        tmp_path.rename(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/evaluation/ego3d_point3r",
                        help="output dir for test.json and scenes.json")
    parser.add_argument("--image-root", default="Ego3D-Bench/Ego3D-Bench/images",
                        help="dir holding the extracted Ego3D-Bench images")
    parser.add_argument("--scenes-root", default="data/media/ego3d/scenes",
                        help="dir to create per-scene symlink dirs in")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N samples (smoke tests)")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("vbdai/Ego3D-Bench")["test"]
    if args.limit is not None:
        dataset = dataset.select(range(args.limit))

    docs, scenes = [], {}
    for sample in dataset:
        docs.append(build_doc(sample))
        scene = build_scene(sample)
        scenes[scene["scene_id"]] = scene

    image_root = Path(args.image_root)
    scenes_root = Path(args.scenes_root)
    for scene in scenes.values():
        link_scene(scene, image_root, scenes_root)

    padded = sum(normalize_scene_resolution(Path(scenes_root) / s["scene_id"])
                 for s in scenes.values())
    print(f"normalized {padded} scenes with mixed view resolutions")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.json").write_text(json.dumps(docs, indent=1))
    (out_dir / "scenes.json").write_text(
        json.dumps(sorted(scenes.values(), key=lambda s: s["scene_id"]), indent=1)
    )
    print(f"wrote {len(docs)} docs and {len(scenes)} scenes to {out_dir}")


if __name__ == "__main__":
    main()
