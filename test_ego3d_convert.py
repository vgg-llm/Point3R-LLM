"""Unit tests for Ego3D-Bench conversion (no network, no GPU)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "scripts" / "preprocess"))

from convert_ego3d import (  # noqa: E402
    EXPECTED_LABELS,
    VIEW_ORDER,
    build_manifest,
    build_prompts,
    parse_view_labels,
    scene_id_of,
    strip_question,
)

WAYMO_QUESTION = (
    "These are five camera views mounted on an ego car\n\n"
    "Front view: <image>\nFront Left view: <image>\nSide Left view: <image>\n"
    "Front Right view: <image>\nSide Right view: <image>\n"
    "The front view corresponds to the north direction. If the ego car moves 2 meters "
    "west while all other objects remain stationary, does the ego car get closer to the "
    "white suv in the front left view?"
)

ARGOVERSE_QUESTION = (
    "These are seven camera views mounted on an ego car\n\n"
    "Front Left view: <image>\nFront view: <image>\nFront Right view: <image>\n"
    "Right view: <image>\nBack Right view: <image>\nBack Left view: <image>\n"
    "Left view: <image>\n"
    "The front-view corresponds to the north direction. If I stand at the location of "
    "blue semi-truck in the front view facing north, is the ego car to my left, right, "
    "front, or back?"
)

WAYMO_SAMPLE = {
    "idx": 7,
    "source": "waymo",
    "category": "Ego_Centric_Motion_Reasoning",
    "question": WAYMO_QUESTION,
    "options": ["A. yes", "B. no"],
    "answer": "A",
    "images": {
        "Front": "seg_FRONT.jpg",
        "Front_Left": "seg_FRONT_LEFT.jpg",
        "Side_Left": "seg_SIDE_LEFT.jpg",
        "Front_Right": "seg_FRONT_RIGHT.jpg",
        "Side_Right": "seg_SIDE_RIGHT.jpg",
        "Back": "",
        "Back_Left": "",
        "Back_Right": "",
    },
}


def test_parse_view_labels_waymo():
    assert parse_view_labels(WAYMO_QUESTION) == [
        "Front", "Front Left", "Side Left", "Front Right", "Side Right",
    ]


def test_parse_view_labels_argoverse_uses_question_wording_not_dict_keys():
    """Argoverse questions say 'Right view'/'Left view' where keys say Side_Right/Side_Left."""
    labels = parse_view_labels(ARGOVERSE_QUESTION)
    assert labels == [
        "Front Left", "Front", "Front Right", "Right", "Back Right", "Back Left", "Left",
    ]
    assert len(labels) == len(VIEW_ORDER["argoverse"])


def test_expected_labels_match_view_order_lengths():
    for source in ("nuscenes", "waymo", "argoverse"):
        assert len(EXPECTED_LABELS[source]) == len(VIEW_ORDER[source])


def test_strip_question_keeps_only_text_after_last_image():
    body = strip_question(WAYMO_QUESTION)
    assert body.startswith("The front view corresponds to the north direction.")
    assert "<image>" not in body
    assert "Front Left view:" not in body


def test_build_manifest_numbers_frames_from_zero():
    manifest = build_manifest(["Front", "Front Left"])
    assert "2 camera views" in manifest
    assert "Frame-0: Front" in manifest
    assert "Frame-1: Front Left" in manifest


def test_build_prompts_pointer_has_pad_and_baseline_does_not():
    pointer, baseline = build_prompts(WAYMO_SAMPLE)
    assert "<|pointer_pad|>" in pointer
    assert "<|pointer_pad|>" not in baseline
    for prompt in (pointer, baseline):
        assert "<image>" not in prompt
        assert "Frame-0: Front" in prompt
        assert "does the ego car get closer" in prompt
        assert prompt.rstrip().endswith("B. no")


def test_build_prompts_rejects_label_mismatch():
    bad = dict(WAYMO_SAMPLE, question=WAYMO_QUESTION.replace("Side Left view", "Left view"))
    with pytest.raises(ValueError, match="label mismatch"):
        build_prompts(bad)


def test_scene_id_is_front_image_stem():
    assert scene_id_of(WAYMO_SAMPLE) == "seg_FRONT"
