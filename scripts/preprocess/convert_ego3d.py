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


def build_manifest(labels):
    lines = [
        f"The scene is provided as {len(labels)} camera views mounted on an ego car, "
        "in this order:"
    ]
    lines += [f"  Frame-{i}: {label}" for i, label in enumerate(labels)]
    return "\n".join(lines)


def build_prompts(sample):
    """Return (pointer_prompt, baseline_prompt).

    The two differ only by the pointer-pad line, so a baseline/pointer comparison
    varies the visual substrate and nothing else.
    """
    source = sample["source"]
    labels = parse_view_labels(sample["question"])
    expected = EXPECTED_LABELS.get(source)
    if expected is not None and labels != expected:
        raise ValueError(
            f"label mismatch for source {source}: parsed {labels}, expected {expected}"
        )

    manifest = build_manifest(labels)
    body = strip_question(sample["question"])
    options = sample.get("options") or []
    tail = "\n".join([body] + list(options))

    baseline_prompt = f"{manifest}\n\n{tail}"
    pointer_prompt = f"{manifest}\n{POINTER_PAD}\n\n{tail}"
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
    source = sample["source"]
    keys = VIEW_ORDER.get(source) or [k for k, v in sample["images"].items() if v]
    names = [sample["images"][key] for key in keys]
    if not all(names):
        raise ValueError(f"sample {sample['idx']} missing an image for source {source}")
    return {"scene_id": scene_id_of(sample), "source": source, "image_names": names}


def _link_names(sample):
    """Return ordered symlink filenames: 00_<View_Key>.jpg, 01_..., preserving suffix."""
    source = sample["source"]
    keys = VIEW_ORDER.get(source) or [k for k, v in sample["images"].items() if v]
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
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(source_path.resolve())
    return scene_dir


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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.json").write_text(json.dumps(docs, indent=1))
    (out_dir / "scenes.json").write_text(
        json.dumps(sorted(scenes.values(), key=lambda s: s["scene_id"]), indent=1)
    )
    print(f"wrote {len(docs)} docs and {len(scenes)} scenes to {out_dir}")


if __name__ == "__main__":
    main()
