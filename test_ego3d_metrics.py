"""Unit tests for the ego3d_point3r lmms_eval task helpers."""
import sys
from pathlib import Path

import pytest

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


LOCALIZATION_OPTIONS = ["A.ego car", "B.white suv", "C.black sedan", "D.blue truck"]


def test_process_results_letter_glued_to_multiword_option_text():
    """'A.ego car' (no space, multi-word option text) must resolve to letter A.

    Uses a doc whose options really are 'A.ego car'-shaped, so the test cannot pass
    just because 'a' happens to equal a yes/no doc's ground truth.
    """
    doc_gt_a = _doc("Localization", "A", "multi_choice", options=LOCALIZATION_OPTIONS)
    row = ego3d.ego3d_process_results(doc_gt_a, ["A.ego car"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    doc_gt_b = _doc("Localization", "B", "multi_choice", options=LOCALIZATION_OPTIONS)
    row = ego3d.ego3d_process_results(doc_gt_b, ["A.ego car"])["ego3d_score"]
    assert row["accuracy"] == 0.0


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
    (the baseline run's think-protocol path), and must be WRONG against GT 'A'."""
    row = ego3d.ego3d_process_results(
        FOUR_OPT_DOC, ["<think>reasoning</think><answer>C</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0  # FOUR_OPT_DOC's ground truth is 'A'
    doc_gt_c = _doc("Object_Centric_Absolute_Distance_MultiChoice", "C", "multi_choice",
                     options=FOUR_OPT_DOC["options"])
    row = ego3d.ego3d_process_results(
        doc_gt_c, ["<think>reasoning</think><answer>C</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0


# --- I3: a leading letter must be a letter, not the first letter of a word --------


def test_free_text_answer_is_not_credited_as_the_option_letter_it_starts_with():
    """'approximately 12 meters' must NOT be read as option A just because it starts
    with an 'a'. Same for 'between ...' -> B and 'cannot ...' -> C."""
    row = ego3d.ego3d_process_results(
        FOUR_OPT_DOC, ["<answer>approximately 12 meters</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0

    travel = _doc("Travel_Time", "B", "multi_choice",
                  options=["A.Less than 5 seconds", "B.5-10 seconds",
                           "C.11-20 seconds", "D.More than 20 seconds"])
    row = ego3d.ego3d_process_results(travel, ["<answer>between 5-10 seconds</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0

    doc_gt_c = _doc("Localization", "C", "multi_choice", options=LOCALIZATION_OPTIONS)
    row = ego3d.ego3d_process_results(doc_gt_c, ["<answer>cannot determine</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


# --- I4: punctuation must not defeat option matching -----------------------------


def test_punctuated_option_text_answer_is_credited():
    """'yes, it is moving toward the ego car' must resolve to letter A via option TEXT."""
    row = ego3d.ego3d_process_results(
        YESNO_DOC_A, ["<answer>yes, it is moving toward the ego car</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    row = ego3d.ego3d_process_results(
        YESNO_DOC_B, ["<answer>yes, it is moving toward the ego car</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


def test_bold_wrapped_option_letter_is_credited():
    """'**B**' must resolve to letter B, and must still be wrong against GT 'A'."""
    row = ego3d.ego3d_process_results(YESNO_DOC_B, ["<answer>**B**</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    row = ego3d.ego3d_process_results(YESNO_DOC_A, ["<answer>**B**</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


def test_trailing_comma_and_paren_forms_are_credited():
    row = ego3d.ego3d_process_results(FOUR_OPT_DOC, ["<answer>A,</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    row = ego3d.ego3d_process_results(FOUR_OPT_DOC, ["<answer>A)</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    row = ego3d.ego3d_process_results(FOUR_OPT_DOC, ["<answer>B)</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


def test_marked_option_letter_inside_a_sentence_is_a_last_resort_only():
    """'The answer is (B).' resolves to B, but a sentence that merely contains the
    article 'a' must not resolve to option A."""
    row = ego3d.ego3d_process_results(
        FOUR_OPT_DOC, ["<answer>The answer is (B).</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0  # GT is 'A'
    doc_gt_b = _doc("Object_Centric_Absolute_Distance_MultiChoice", "B", "multi_choice",
                     options=FOUR_OPT_DOC["options"])
    row = ego3d.ego3d_process_results(
        doc_gt_b, ["<answer>The answer is (B).</answer>"])["ego3d_score"]
    assert row["accuracy"] == 1.0
    row = ego3d.ego3d_process_results(
        FOUR_OPT_DOC, ["<answer>there is a black suv in the front view</answer>"])["ego3d_score"]
    assert row["accuracy"] == 0.0


# --- C2: numeric answers must come from a real answer, not from the reasoning ----


def test_extract_number_does_not_read_minus_one_out_of_a_frame_reference():
    """The manifest names frames 'Frame-1' / '<frame-1>'; neither may be read as -1."""
    assert ego3d.extract_number("Frame-1") is None
    assert ego3d.extract_number("<answer>Frame-1</answer>") is None
    assert ego3d.extract_number("<answer><frame-3></answer>") is None
    # Even in the short protocol, where untagged text IS read, the frame reference
    # must not contribute a negative number.
    assert ego3d.extract_number("closest in Frame-2, about 7 m") == 7.0
    row = ego3d.ego3d_process_results(NUM_DOC, ["Frame-1"])["ego3d_score"]
    assert row["squared_error"] == (100.0 - 13.7) ** 2  # no answer -> worst case


TRUNCATED_REASONING = (
    "In Frame-1, we see a black SUV about 10 m away on the left side of the road, "
    "and the white delivery truck is further ahead, so the distance between them"
)


def test_extract_number_returns_none_for_unfinished_think_reasoning():
    """A think-protocol response truncated before any <answer> must not be mined:
    the number must come from the answer, not from whatever the prose mentioned."""
    assert ego3d.extract_number(TRUNCATED_REASONING, require_answer_tag=True) is None
    row = ego3d.ego3d_process_results_think(NUM_DOC, [TRUNCATED_REASONING])["ego3d_score"]
    assert row["squared_error"] == (100.0 - 13.7) ** 2


def test_extract_number_returns_none_for_unclosed_answer_tag_under_both_protocols():
    unclosed = "<think>reasoning</think><answer>13.7"
    assert ego3d.extract_number(unclosed) is None
    assert ego3d.extract_number(unclosed, require_answer_tag=True) is None
    for scorer in (ego3d.ego3d_process_results, ego3d.ego3d_process_results_think):
        row = scorer(NUM_DOC, [unclosed])["ego3d_score"]
        assert row["squared_error"] == (100.0 - 13.7) ** 2


def test_think_scorer_reads_a_properly_closed_answer_span():
    row = ego3d.ego3d_process_results_think(
        NUM_DOC, ["<think>reasoning about Frame-1</think><answer>13.7</answer>"])["ego3d_score"]
    assert row["squared_error"] == 0.0


def test_extract_number_still_reads_short_untagged_answers():
    """The short protocol answers in a word or phrase, with no tags at all."""
    assert ego3d.extract_number("13.7") == 13.7
    assert ego3d.extract_number("about 8 meters") == 8.0
    assert ego3d.extract_number("approximately 12.5 m from the ego car") == 12.5
    row = ego3d.ego3d_process_results(NUM_DOC, ["13.7 meters"])["ego3d_score"]
    assert row["squared_error"] == 0.0


TASK_DIR = Path(__file__).parent / "src" / "lmms_eval" / "tasks" / "ego3d_point3r"


def test_think_protocol_tasks_wire_the_strict_scorer():
    """The two <think>/<answer> tasks must use ego3d_process_results_think; the short
    task must not. Getting this wrong silently reopens the prose-mining path."""
    for name in ("ego3d_baseline.yaml", "ego3d_point3r_think.yaml"):
        text = (TASK_DIR / name).read_text()
        assert "process_results: !function utils.ego3d_process_results_think" in text, name
        assert "protocol: \"think\"" in text, name
    short = (TASK_DIR / "ego3d_point3r.yaml").read_text()
    assert "ego3d_process_results_think" not in short
    assert "protocol: \"short\"" in short
    # The shared default supplies the short-protocol scorer that the short task uses.
    assert "process_results: !function utils.ego3d_process_results" in (
        TASK_DIR / "ego3d_default_yaml").read_text()


def _load_docs():
    path = Path(__file__).parent / "data" / "evaluation" / "ego3d_point3r" / "test.json"
    if not path.exists():
        pytest.skip(f"converted docs not present: {path}")
    import json
    return json.loads(path.read_text())


def test_floor_tables_match_the_ground_truth_they_claim_to_summarize():
    """Recompute every floor from the converted ground truth, so a wrong number in
    the tables fails here instead of silently flattering a result."""
    docs = _load_docs()
    assert len(docs) == 8675

    by_category = {}
    for doc in docs:
        by_category.setdefault(doc["metadata"]["category"], []).append(doc)
    assert len(by_category) == 10
    assert set(ego3d.CHANCE_FLOORS) == set(by_category) - ego3d.EXACT_NUMBER_CATEGORIES
    assert set(ego3d.MAJORITY_FLOORS) == set(ego3d.CHANCE_FLOORS)
    assert set(ego3d.CONSTANT_RMSE_FLOORS) == ego3d.EXACT_NUMBER_CATEGORIES

    for category, rows in by_category.items():
        answers = [row["answer"] for row in rows]
        if category in ego3d.EXACT_NUMBER_CATEGORIES:
            values = [float(a) for a in answers]
            mean = sum(values) / len(values)
            rmse = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            assert round(rmse, 1) == ego3d.CONSTANT_RMSE_FLOORS[category], category
        else:
            counts = {}
            for answer in answers:
                counts[answer] = counts.get(answer, 0) + 1
            option_counts = {len(row["options"]) for row in rows}
            assert len(option_counts) == 1, (category, option_counts)
            chance = 100.0 / option_counts.pop()
            assert round(chance, 1) == ego3d.CHANCE_FLOORS[category], category
            majority = 100.0 * max(counts.values()) / len(answers)
            assert round(majority, 1) == ego3d.MAJORITY_FLOORS[category], category
