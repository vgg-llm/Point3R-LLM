"""Unit tests for the ego3d_point3r lmms_eval task helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "lmms_eval" / "tasks" / "ego3d_point3r"))

import utils as ego3d  # noqa: E402


def _doc(category, answer, question_type):
    return {
        "conversations": [{"from": "human", "value": "MANIFEST\n<|pointer_pad|>\n\nQ?\nA. yes\nB. no"},
                          {"from": "gpt", "value": answer}],
        "answer": answer,
        "baseline_prompt": "MANIFEST\n\nQ?\nA. yes\nB. no",
        "pointer_data": "ego3d/pointer_memory_qwen3vl/s.pt",
        "images": ["ego3d/scenes/s/00_Front.jpg"],
        "metadata": {"idx": 1, "source": "waymo", "category": category,
                     "scene_id": "s", "question_type": question_type},
    }


MC_DOC = _doc("Localization", "C", "multi_choice")
NUM_DOC = _doc("Ego_Centric_Absolute_Distance", "13.7", "exact_number")


def test_doc_to_text_short_protocol_uses_pointer_prompt_and_letter_suffix():
    text = ego3d.ego3d_doc_to_text(MC_DOC, {"protocol": "short", "visual_mode": "pointer"})
    assert "<|pointer_pad|>" in text
    assert text.endswith("Answer with the option's letter from the given choices directly.")


def test_doc_to_text_images_mode_uses_baseline_prompt():
    text = ego3d.ego3d_doc_to_text(MC_DOC, {"protocol": "think", "visual_mode": "images"})
    assert "<|pointer_pad|>" not in text


def test_doc_to_text_think_protocol_picks_suffix_by_category():
    number = ego3d.ego3d_doc_to_text(NUM_DOC, {"protocol": "think", "visual_mode": "pointer"})
    assert "final answer (number only)" in number
    yesno = ego3d.ego3d_doc_to_text(
        _doc("Object_Centric_Motion_Reasoning", "A", "multi_choice"),
        {"protocol": "think", "visual_mode": "pointer"})
    assert "final answer (yes or no)" in yesno
    letter = ego3d.ego3d_doc_to_text(MC_DOC, {"protocol": "think", "visual_mode": "pointer"})
    assert "only the letter of the choice" in letter


def test_doc_to_text_short_protocol_numeric_suffix():
    text = ego3d.ego3d_doc_to_text(NUM_DOC, {"protocol": "short", "visual_mode": "pointer"})
    assert text.endswith("Please answer the question using a single word or phrase.")


def test_extract_answer_handles_both_protocols():
    assert ego3d.extract_answer("<think>blah</think><answer>C</answer>") == "c"
    assert ego3d.extract_answer("C. Right") == "c"
    assert ego3d.extract_answer(" B.\n") == "b"


def test_extract_number_takes_first_number():
    assert ego3d.extract_number("<answer>13.7</answer>") == 13.7
    assert ego3d.extract_number("about 8 meters") == 8.0
    assert ego3d.extract_number("no idea") is None


def test_process_results_scores_multi_choice():
    correct = ego3d.ego3d_process_results(MC_DOC, ["<answer>C</answer>"])["ego3d_score"]
    assert correct["accuracy"] == 1.0
    wrong = ego3d.ego3d_process_results(MC_DOC, ["<answer>A</answer>"])["ego3d_score"]
    assert wrong["accuracy"] == 0.0


def test_process_results_squared_error_and_clipping():
    exact = ego3d.ego3d_process_results(NUM_DOC, ["13.7"])["ego3d_score"]
    assert exact["squared_error"] == 0.0
    clipped = ego3d.ego3d_process_results(NUM_DOC, ["500"])["ego3d_score"]
    assert clipped["squared_error"] == (100.0 - 13.7) ** 2


def test_unparseable_number_scores_worst_case_not_dropped():
    """Deliberate deviation from upstream, which drops these from the RMSE."""
    row = ego3d.ego3d_process_results(NUM_DOC, ["I cannot tell"])["ego3d_score"]
    assert row["squared_error"] == (100.0 - 13.7) ** 2


def test_aggregate_reports_per_category_scores_rmse_and_floors():
    rows = [
        ego3d.ego3d_process_results(MC_DOC, ["<answer>C</answer>"])["ego3d_score"],
        ego3d.ego3d_process_results(MC_DOC, ["<answer>A</answer>"])["ego3d_score"],
        ego3d.ego3d_process_results(NUM_DOC, ["13.7"])["ego3d_score"],
        ego3d.ego3d_process_results(NUM_DOC, ["15.7"])["ego3d_score"],
    ]
    overall = ego3d.ego3d_aggregate_results(rows)
    assert overall == 50.0  # one of two multi-choice correct, in percent


def test_floor_tables_cover_all_ten_categories():
    assert len(ego3d.CHANCE_FLOORS) == 8
    assert len(ego3d.MAJORITY_FLOORS) == 8
    assert set(ego3d.CONSTANT_RMSE_FLOORS) == ego3d.EXACT_NUMBER_CATEGORIES
    assert ego3d.CONSTANT_RMSE_FLOORS["Ego_Centric_Absolute_Distance"] == 8.4
