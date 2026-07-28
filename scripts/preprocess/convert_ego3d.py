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
