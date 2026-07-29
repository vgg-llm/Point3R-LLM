"""Unit tests for the ego3d_point3r lmms_eval task helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "lmms_eval" / "tasks" / "ego3d_point3r"))

import utils as ego3d  # noqa: E402


def _doc(category, answer, question_type, options=None):
    doc = {
        "conversations": [{"from": "human", "value": "MANIFEST\n<|pointer_pad|>\n\nQ?\nA. yes\nB. no"},
                          {"from": "gpt", "value": answer}],
        "answer": answer,
        "baseline_prompt": "MANIFEST\n\nQ?\nA. yes\nB. no",
        "pointer_data": "ego3d/pointer_memory_qwen3vl/s.pt",
        "images": ["ego3d/scenes/s/00_Front.jpg"],
        "metadata": {"idx": 1, "source": "waymo", "category": category,
                     "scene_id": "s", "question_type": question_type},
    }
    if options is not None:
        doc["options"] = options
    return doc


MC_DOC = _doc("Localization", "C", "multi_choice")
NUM_DOC = _doc("Ego_Centric_Absolute_Distance", "13.7", "exact_number")

YESNO_DOC_A = _doc("Object_Centric_Motion_Reasoning", "A", "multi_choice",
                    options=["A. yes", "B. no"])
YESNO_DOC_B = _doc("Ego_Centric_Motion_Reasoning", "B", "multi_choice",
                    options=["A. yes", "B. no"])


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


def test_process_results_credits_yes_answer_text_against_letter_a_ground_truth():
    """Deviation 3: GT is the letter 'A', options are text-valued ('A. yes'/'B. no'),
    and the model (correctly) answers with the option text, not the letter."""
    row = ego3d.ego3d_process_results(YESNO_DOC_A, ["<answer>yes</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_credits_no_answer_text_against_letter_b_ground_truth():
    row = ego3d.ego3d_process_results(YESNO_DOC_B, ["<answer>no</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_still_scores_wrong_when_text_answer_mismatches_ground_truth():
    """'yes' resolves to letter 'A', which is wrong against GT 'B'."""
    row = ego3d.ego3d_process_results(YESNO_DOC_B, ["<answer>yes</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


def test_process_results_plain_letter_prediction_unchanged():
    """A model that already answers with the letter must keep scoring exactly as before."""
    correct = ego3d.ego3d_process_results(YESNO_DOC_A, ["<answer>A</answer>"])["ego3d_score"]
    assert correct["accuracy"] == 1.0
    wrong = ego3d.ego3d_process_results(YESNO_DOC_A, ["<answer>B</answer>"])["ego3d_score"]
    assert wrong["accuracy"] == 0.0


def test_process_results_no_options_key_behaves_as_before():
    """MC_DOC carries no 'options' key at all (mirrors docs predating this fix /
    categories with no text-valued options); option-text resolution must be a no-op."""
    correct = ego3d.ego3d_process_results(MC_DOC, ["<answer>C</answer>"])["ego3d_score"]
    assert correct["accuracy"] == 1.0
    text_answer = ego3d.ego3d_process_results(MC_DOC, ["<answer>yes</answer>"])["ego3d_score"]
    assert text_answer["accuracy"] == 0.0


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


FOUR_OPT_DOC = _doc(
    "Object_Centric_Absolute_Distance_MultiChoice", "A", "multi_choice",
    options=["A.36 meters", "B.25 meters", "C.13 meters", "D.3 meters"],
)


def test_process_results_letter_glued_to_option_text_no_separator():
    """'A.1 meter' (no space after the marker) must resolve to letter A."""
    row = ego3d.ego3d_process_results(FOUR_OPT_DOC, ["A.1 meter"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    wrong_gt = _doc("Object_Centric_Absolute_Distance_MultiChoice", "B", "multi_choice",
                     options=FOUR_OPT_DOC["options"])
    row = ego3d.ego3d_process_results(wrong_gt, ["A.1 meter"])["ego3d_score"]
    assert row["accuracy"] == 0.0


def test_process_results_letter_glued_to_multiword_option_text():
    """'A.ego car' (no space, multi-word option text) must resolve to letter A."""
    row = ego3d.ego3d_process_results(YESNO_DOC_A, ["A.ego car"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_letter_with_space_after_marker_regression():
    """'A. yes' (space after marker, the format that already worked) must keep working."""
    row = ego3d.ego3d_process_results(YESNO_DOC_A, ["A. yes"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_option_text_only_still_resolves_via_text_path():
    """Bare 'yes' (no letter at all) must still resolve via option-TEXT matching."""
    row = ego3d.ego3d_process_results(YESNO_DOC_A, ["yes"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_no_not_misparsed_as_letter_n():
    """'no' must not be misread as option letter N; it must resolve to letter B via
    option-TEXT matching against GT 'B'."""
    row = ego3d.ego3d_process_results(YESNO_DOC_B, ["no"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_letter_with_colon_separator():
    """'D: 4 meters' (colon separator) must resolve to letter D."""
    doc = _doc("Travel_Time", "D", "multi_choice",
                options=["A.Less than 5 seconds", "B.5-10 seconds",
                         "C.11-20 seconds", "D.More than 20 seconds"])
    row = ego3d.ego3d_process_results(doc, ["D: 4 meters"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_process_results_think_protocol_bare_letter_regression():
    """Bare '<answer>C</answer>' with 4 options must still resolve to letter C
    (the baseline run's think-protocol path)."""
    row = ego3d.ego3d_process_results(FOUR_OPT_DOC, ["<think>reasoning</think><answer>C</answer>"])["ego3d_score"]
    doc_gt_c = _doc("Object_Centric_Absolute_Distance_MultiChoice", "C", "multi_choice",
                     options=FOUR_OPT_DOC["options"])
    row = ego3d.ego3d_process_results(doc_gt_c, ["<think>reasoning</think><answer>C</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0


def test_floor_tables_cover_all_ten_categories():
    assert len(ego3d.CHANCE_FLOORS) == 8
    assert len(ego3d.MAJORITY_FLOORS) == 8
    assert set(ego3d.CONSTANT_RMSE_FLOORS) == ego3d.EXACT_NUMBER_CATEGORIES
    assert ego3d.CONSTANT_RMSE_FLOORS["Ego_Centric_Absolute_Distance"] == 8.4
