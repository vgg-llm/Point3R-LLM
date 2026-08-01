"""Unit tests for Ego3D-Bench conversion (no network, no GPU)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "scripts" / "preprocess"))

from convert_ego3d import (  # noqa: E402
    EXPECTED_LABELS,
    POINTER_FRAME_LABEL,
    POINTER_INDEX_BASE,
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


def test_build_manifest_numbers_frames_from_zero_by_default():
    """Default style is the IMAGES path's: 0-indexed 'Frame-N', no brackets."""
    manifest = build_manifest(["Front", "Front Left"])
    assert "2 camera views" in manifest
    assert "Frame-0: Front" in manifest
    assert "Frame-1: Front Left" in manifest
    assert "<frame" not in manifest


def test_build_manifest_pointer_style_is_one_indexed_and_bracketed():
    """The POINTER path emits '<frame-1>', '<frame-2>', ... (pointer_data.py:195)."""
    manifest = build_manifest(
        ["Front", "Front Left"],
        index_base=POINTER_INDEX_BASE, label_template=POINTER_FRAME_LABEL,
    )
    assert "<frame-1>: Front" in manifest
    assert "<frame-2>: Front Left" in manifest
    assert "frame-0" not in manifest
    assert "Frame-" not in manifest


def test_build_prompts_pointer_manifest_matches_pointer_token_labels():
    """C1: the pointer manifest must name <frame-1>..<frame-N>, matching the token
    groups pointer_data.py emits; the baseline manifest must keep Frame-0..N-1."""
    pointer, baseline = build_prompts(WAYMO_SAMPLE)
    labels = ["Front", "Front Left", "Side Left", "Front Right", "Side Right"]

    for i, label in enumerate(labels):
        assert f"<frame-{i + 1}>: {label}" in pointer
        assert f"Frame-{i}: {label}" in baseline
    # No off-by-one leftovers in either direction.
    assert "<frame-0>" not in pointer
    assert f"<frame-{len(labels) + 1}>" not in pointer
    assert "Frame-" not in pointer
    assert f"Frame-{len(labels)}:" not in baseline
    assert "<frame" not in baseline


def test_build_prompts_pointer_has_pad_and_baseline_does_not():
    pointer, baseline = build_prompts(WAYMO_SAMPLE)
    assert "<|pointer_pad|>" in pointer
    assert "<|pointer_pad|>" not in baseline
    for prompt in (pointer, baseline):
        assert "<image>" not in prompt
        assert "does the ego car get closer" in prompt
        assert prompt.rstrip().endswith("B. no")


def test_build_prompts_rejects_label_mismatch():
    bad = dict(WAYMO_SAMPLE, question=WAYMO_QUESTION.replace("Side Left view", "Left view"))
    with pytest.raises(ValueError, match="label mismatch"):
        build_prompts(bad)


def test_scene_id_is_front_image_stem():
    assert scene_id_of(WAYMO_SAMPLE) == "seg_FRONT"


from convert_ego3d import build_doc, build_scene, link_scene  # noqa: E402


def test_build_doc_schema():
    doc = build_doc(WAYMO_SAMPLE)
    assert doc["answer"] == "A"
    assert doc["conversations"][0]["from"] == "human"
    assert "<|pointer_pad|>" in doc["conversations"][0]["value"]
    assert doc["conversations"][1]["value"] == "A"
    assert "<|pointer_pad|>" not in doc["baseline_prompt"]
    assert doc["pointer_data"] == "ego3d/pointer_memory_qwen3vl/seg_FRONT.pt"
    assert doc["images"][0] == "ego3d/scenes/seg_FRONT/00_Front.jpg"
    assert len(doc["images"]) == 5
    assert doc["metadata"]["question_type"] == "multi_choice"
    assert doc["metadata"]["scene_id"] == "seg_FRONT"


def test_build_doc_carries_options():
    doc = build_doc(WAYMO_SAMPLE)
    assert doc["options"] == ["A. yes", "B. no"]


def test_build_doc_options_defaults_to_empty_list_when_absent():
    sample = dict(WAYMO_SAMPLE, category="Ego_Centric_Absolute_Distance",
                  options=None, answer="13.7")
    doc = build_doc(sample)
    assert doc["options"] == []


def test_build_doc_marks_absolute_distance_as_exact_number():
    sample = dict(WAYMO_SAMPLE, category="Ego_Centric_Absolute_Distance",
                  options=None, answer="13.7")
    doc = build_doc(sample)
    assert doc["metadata"]["question_type"] == "exact_number"
    assert doc["answer"] == "13.7"


def test_build_scene_orders_names_canonically():
    scene = build_scene(WAYMO_SAMPLE)
    assert scene["scene_id"] == "seg_FRONT"
    assert scene["image_names"] == [
        "seg_FRONT.jpg", "seg_FRONT_LEFT.jpg", "seg_SIDE_LEFT.jpg",
        "seg_FRONT_RIGHT.jpg", "seg_SIDE_RIGHT.jpg",
    ]


def test_link_scene_creates_ordered_symlinks(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in build_scene(WAYMO_SAMPLE)["image_names"]:
        (image_root / name).write_bytes(b"fake")

    scenes_root = tmp_path / "scenes"
    scene_dir = link_scene(build_scene(WAYMO_SAMPLE), image_root, scenes_root)

    linked = sorted(p.name for p in scene_dir.iterdir())
    assert linked == [
        "00_Front.jpg", "01_Front_Left.jpg", "02_Side_Left.jpg",
        "03_Front_Right.jpg", "04_Side_Right.jpg",
    ]
    assert all((scene_dir / name).is_symlink() for name in linked)
    assert (scene_dir / "00_Front.jpg").resolve() == (image_root / "seg_FRONT.jpg").resolve()


def test_unknown_source_fails_identically_in_build_scene_and_link_names(tmp_path):
    """build_scene, _link_names and link_scene all index VIEW_ORDER directly, so an
    unrecognized source raises in all three rather than half-succeeding in some."""
    from convert_ego3d import _link_names

    bad = dict(WAYMO_SAMPLE, source="kitti")
    with pytest.raises(KeyError):
        build_scene(bad)
    with pytest.raises(KeyError):
        _link_names(bad)
    with pytest.raises(KeyError):
        link_scene({"source": "kitti", "scene_id": "s", "image_names": []},
                   tmp_path / "images", tmp_path / "scenes")


def test_link_scene_fails_loudly_on_missing_image(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    with pytest.raises(FileNotFoundError, match="seg_FRONT.jpg"):
        link_scene(build_scene(WAYMO_SAMPLE), image_root, tmp_path / "scenes")


from convert_ego3d import normalize_scene_resolution  # noqa: E402


def _write_jpeg(path, size):
    from PIL import Image
    Image.new("RGB", size, (128, 128, 128)).save(path, "JPEG")


def test_normalize_pads_mixed_resolution_scene_in_place(tmp_path):
    from PIL import Image
    scene = tmp_path / "waymo_scene"
    scene.mkdir()
    _write_jpeg(scene / "00_Front.jpg", (1920, 1280))
    _write_jpeg(scene / "01_Side_Left.jpg", (1920, 886))

    assert normalize_scene_resolution(scene) is True

    sizes = {p.name: Image.open(p).size for p in sorted(scene.iterdir())}
    assert sizes == {"00_Front.jpg": (1920, 1280), "01_Side_Left.jpg": (1920, 1280)}
    # Padding is centered and black, so the top row of the padded view is black.
    padded = Image.open(scene / "01_Side_Left.jpg")
    assert padded.getpixel((960, 2)) == (0, 0, 0)
    # And the original content survives in the middle.
    assert padded.getpixel((960, 640)) != (0, 0, 0)


def test_normalize_leaves_uniform_scene_untouched(tmp_path):
    scene = tmp_path / "nuscenes_scene"
    scene.mkdir()
    _write_jpeg(scene / "00_Front.jpg", (1600, 900))
    _write_jpeg(scene / "01_Back.jpg", (1600, 900))
    before = {p.name: p.read_bytes() for p in scene.iterdir()}

    assert normalize_scene_resolution(scene) is False

    after = {p.name: p.read_bytes() for p in scene.iterdir()}
    assert after == before


def test_normalize_is_idempotent(tmp_path):
    scene = tmp_path / "waymo_scene"
    scene.mkdir()
    _write_jpeg(scene / "00_Front.jpg", (1920, 1280))
    _write_jpeg(scene / "01_Side_Left.jpg", (1920, 886))

    assert normalize_scene_resolution(scene) is True
    assert normalize_scene_resolution(scene) is False


def test_normalize_replaces_symlink_with_real_file(tmp_path):
    real = tmp_path / "real_side.jpg"
    _write_jpeg(real, (1920, 886))
    scene = tmp_path / "waymo_scene"
    scene.mkdir()
    _write_jpeg(scene / "00_Front.jpg", (1920, 1280))
    (scene / "01_Side_Left.jpg").symlink_to(real)

    normalize_scene_resolution(scene)

    assert not (scene / "01_Side_Left.jpg").is_symlink()
    from PIL import Image
    assert Image.open(real).size == (1920, 886)  # source image untouched


def test_link_scene_does_not_clobber_already_padded_real_file(tmp_path):
    """A view path that already holds a real (padded) JPEG must survive a re-run of
    link_scene untouched, while paths that are still symlinks get (re)created as usual.

    This reproduces the normalize -> re-link -> re-normalize churn: link_scene used to
    unconditionally unlink+recreate every view as a symlink to the original source,
    which destroyed padding done by normalize_scene_resolution on every re-run.
    """
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in build_scene(WAYMO_SAMPLE)["image_names"]:
        (image_root / name).write_bytes(b"fake-source")

    scenes_root = tmp_path / "scenes"
    scene = build_scene(WAYMO_SAMPLE)
    scene_dir = link_scene(scene, image_root, scenes_root)

    # Simulate normalize_scene_resolution having padded one view in place: replace its
    # symlink with a real file holding different (padded) bytes.
    padded_path = scene_dir / "01_Front_Left.jpg"
    assert padded_path.is_symlink()
    padded_path.unlink()
    padded_path.write_bytes(b"padded-real-file")

    # Re-run link_scene, as main() does on every converter invocation.
    link_scene(scene, image_root, scenes_root)

    # The padded real file must be left completely alone.
    assert not padded_path.is_symlink()
    assert padded_path.read_bytes() == b"padded-real-file"

    # Every other view must still be a symlink, recreated as normal.
    for name in ["00_Front.jpg", "02_Side_Left.jpg", "03_Front_Right.jpg", "04_Side_Right.jpg"]:
        assert (scene_dir / name).is_symlink()
