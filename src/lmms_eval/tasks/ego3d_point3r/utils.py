"""Ego3D-Bench evaluation for Point3R-LLM.

Ports Ego3D-Bench/utils/eval.py: lowercased exact match for the 8 multi-choice
categories, RMSE (predictions clipped to 100 m) for the 2 absolute-distance ones.

Deliberate deviations from upstream:
  1. Unparseable numeric predictions score worst-case instead of being dropped
     from the RMSE. Upstream's `if pred:` also drops legitimate 0 predictions.
  2. No resume logic; lmms_eval owns sharding and logging.
  3. Predictions given as option TEXT (e.g. "yes") rather than option LETTER are
     credited by resolving them against the doc's own options, generically (not
     just yes/no). Upstream's own scorer has this same yes/no-versus-letter gap
     for the three categories whose ground truth is a letter but whose options
     and prompt suffix are text-valued ("A. yes" / "B. no").
  4. Numeric predictions are clipped to [0, 100]; upstream only upper-clips at 100.
  5. A leading option letter is recognized even when glued to the option text
     with no separating space (e.g. "A.1 meter", "A.ego car", "B)", "D: 4 meters"),
     not just when cleanly separated ("A. yes"). Guarded by the doc's own option
     count so a text answer like "no"/"yes" is never misread as letter N/Y;
     those still resolve through the option-TEXT matching in deviation 3.
     Upstream's own scorer takes only the first whitespace token and has the
     same glued-letter gap, so our port is strictly more permissive in what it
     credits here. The leading letter must END the token or be followed by a
     non-letter, so free text ("approximately 12 meters", "cannot determine") is
     never credited as option A / C. Wrapping punctuation is stripped from both
     ends of the token and the span, so "**B**" and "yes, it is moving ..." are
     matched as "b" and "yes".
  6. A numeric answer is only read from a properly CLOSED <answer></answer> span,
     or from an untagged response short enough to be the "single word or phrase"
     the short protocol asks for. A think-protocol response truncated at the
     generation cap therefore scores the documented worst case (deviation 1)
     rather than having a number mined out of its reasoning -- where the number
     regex would also have matched the `-1` inside a "Frame-1" reference.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

# metadata lives in the shared ego3d_default_yaml that all three task yamls include.
with open(Path(__file__).parent / "ego3d_default_yaml", "r") as f:
    _raw = [line for line in f.readlines() if "!function" not in line]
media_dir = yaml.safe_load("".join(_raw))["metadata"]["media_dir"]

EXACT_NUMBER_CATEGORIES = {
    "Ego_Centric_Absolute_Distance",
    "Object_Centric_Absolute_Distance",
}

YES_NO_CATEGORIES = {
    "Ego_Centric_Relative_Distance",
    "Ego_Centric_Motion_Reasoning",
    "Object_Centric_Motion_Reasoning",
}

MAX_DISTANCE_M = 100.0

# Trivial-baseline floors, computed once from the 8675 ground-truth answers.
CHANCE_FLOORS = {
    "Ego_Centric_Relative_Distance": 50.0,
    "Ego_Centric_Motion_Reasoning": 50.0,
    "Object_Centric_Motion_Reasoning": 50.0,
    "Object_Centric_Relative_Distance": 50.0,
    "Localization": 25.0,
    "Travel_Time": 25.0,
    "Ego_Centric_Absolute_Distance_MultiChoice": 25.0,
    "Object_Centric_Absolute_Distance_MultiChoice": 25.0,
}

MAJORITY_FLOORS = {
    "Ego_Centric_Relative_Distance": 55.1,
    "Ego_Centric_Motion_Reasoning": 62.1,
    "Object_Centric_Motion_Reasoning": 57.7,
    "Object_Centric_Relative_Distance": 63.6,
    "Localization": 36.5,
    "Travel_Time": 36.2,
    "Ego_Centric_Absolute_Distance_MultiChoice": 26.3,
    "Object_Centric_Absolute_Distance_MultiChoice": 27.0,
}

# RMSE of always answering the GT mean. Beats every open-source baseline in the
# paper, so report it next to any RMSE we produce.
CONSTANT_RMSE_FLOORS = {
    "Ego_Centric_Absolute_Distance": 8.4,
    "Object_Centric_Absolute_Distance": 10.2,
}

THINK_SUFFIX_NUMBER = (
    "\nOutput the thinking process in <think> </think> and final answer "
    "(number only) in <answer> </answer> tags."
)
THINK_SUFFIX_YESNO = (
    "\nOutput the thinking process in <think> </think> and final answer "
    "(yes or no) in <answer> </answer> tags."
)
THINK_SUFFIX_LETTER = (
    "\nOutput the thinking process in <think> </think> and final answer "
    "(only the letter of the choice) in <answer> </answer> tags."
)
SHORT_SUFFIX_NUMBER = "\nPlease answer the question using a single word or phrase."
SHORT_SUFFIX_LETTER = "\nAnswer with the option's letter from the given choices directly."


def _category(doc):
    return doc["metadata"]["category"]


def ego3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    protocol = kwargs.get("protocol", "short")
    visual_mode = kwargs.get("visual_mode", "pointer")
    category = _category(doc)

    prompt = (
        doc["conversations"][0]["value"] if visual_mode == "pointer"
        else doc["baseline_prompt"]
    )

    if protocol == "think":
        if category in EXACT_NUMBER_CATEGORIES:
            suffix = THINK_SUFFIX_NUMBER
        elif category in YES_NO_CATEGORIES:
            suffix = THINK_SUFFIX_YESNO
        else:
            suffix = THINK_SUFFIX_LETTER
    else:
        suffix = (
            SHORT_SUFFIX_NUMBER if category in EXACT_NUMBER_CATEGORIES
            else SHORT_SUFFIX_LETTER
        )
    return prompt + suffix


def ego3d_doc_to_visual_pointer(doc):
    """Pointer mode supplies no visuals; the scene arrives as pointer tokens."""
    return [[]]


def ego3d_doc_to_visual_images(doc):
    """Load the scene's views as PIL images, in canonical order."""
    images = [
        Image.open(os.path.join(media_dir, rel)).convert("RGB")
        for rel in doc["images"]
    ]
    return [images]


# Punctuation a model wraps or trails its answer with: "yes,", "**B**", "(C)", "B."
_ANSWER_PUNCTUATION = ".,;:!*)\"'"


def _closed_answer_span(text):
    """Return the content of a properly closed <answer>...</answer> span, else None.

    Deviation 6: an OPEN-but-unclosed `<answer>` is treated as no answer at all. Under
    the think protocol a response truncated at the generation cap has neither a closed
    span nor a usable answer, and reading whatever follows the opening tag (or, worse,
    the reasoning that precedes it) mines prose instead of scoring an answer.
    """
    if "<answer>" not in text:
        return None
    start = text.find("<answer>") + len("<answer>")
    end = text.find("</answer>", start)
    if end == -1:
        return None
    return text[start:end]


def _answer_span(text):
    """Return the closed <answer> span, else the text after an unclosed opening tag,
    else the whole text.

    Multi-choice matching stays lenient about a missing closing tag (a truncated
    "<answer>C" still names an option); only numeric extraction requires a closed
    span, because there the fallback is prose-mining rather than a single token.
    """
    span = _closed_answer_span(text)
    if span is not None:
        return span
    if "<answer>" in text:
        return text[text.find("<answer>") + len("<answer>"):]
    return text


def _strip_answer_punctuation(token):
    """Strip wrapping/trailing punctuation from both ends ("**B**" -> "B", "yes," -> "yes")."""
    return token.strip().strip(_ANSWER_PUNCTUATION).strip()


def extract_answer(text):
    """Return the comparable multi-choice answer, lowercased.

    Protocol-agnostic: reads the <answer> span when present, else the first token.
    Surrounding punctuation is stripped from both ends, so "**B**", "yes," and "B."
    all reduce to their bare token.
    """
    text = _answer_span(text)
    token = text.replace("\n", " ").strip().split(" ")[0]
    return _strip_answer_punctuation(token).lower()


def _full_answer_span(text):
    """Return the whole (whitespace-normalized) answer span, lowercased.

    Unlike extract_answer, this keeps multi-word spans intact so option text such
    as "yes" or, in principle, multi-word option text can be matched in full.
    Surrounding punctuation is stripped from both ends.
    """
    text = _answer_span(text)
    normalized = " ".join(text.replace("\n", " ").strip().split())
    return _strip_answer_punctuation(normalized).lower()


_OPTION_LETTER_RE = re.compile(r"^\s*([A-Za-z])\s*[.)-]?\s*(.*)$")

# A leading option letter that is not merely the first letter of a word: the letter
# must end the token or be followed by a non-letter. Credits "a" (bare), "a.1",
# "a.ego car", "b)", "d:", "a," -- the letter/separator forms the benchmark's own
# option strings and the models produce -- while refusing "approximately",
# "between", "cannot", which would otherwise be read as options A, B and C.
_LEADING_OPTION_LETTER_RE = re.compile(r"^([a-z])(?=$|[^a-z])")

# Last-resort option-letter recognition inside a longer answer span: the letter must
# be explicitly marked up ("(b)", "**b**", "[c]") or be the very last thing in the
# span ("the answer is b."). Deliberately does NOT match a letter merely occurring as
# a word, so the article "a" in "a black suv is closer" is never read as option A.
_MARKED_OPTION_LETTER_RES = (
    re.compile(r"[(\[*]\s*([a-z])\s*[)\]*.:,]"),
    re.compile(r"(?:^|[\s:*(\[])([a-z])[.)\]*]?$"),
)


def _resolve_option_letter(prediction, options):
    """Return the lowercase option letter the prediction leads with, or None.

    Guarded by the doc's own option count: a leading character is only accepted
    as an option letter if it falls within `A..<len(options)>` for this doc.
    Without that guard, a text answer like "no" would be misread as letter `N`
    and "yes" as `Y`; those must fall through to option-TEXT matching instead
    (see `_option_text_to_letter`).
    """
    num_options = len(options) if options else 0
    if num_options == 0:
        return None
    token = extract_answer(prediction)
    match = _LEADING_OPTION_LETTER_RE.match(token)
    if not match:
        return None
    letter = match.group(1)
    if ord(letter) - ord("a") < num_options:
        return letter
    return None


def _option_text_to_letter(options):
    """Map each option's lowercased text (letter marker stripped) to its lowercased letter.

    "A. yes" -> {"yes": "a"}, "B.no" -> {"no": "b"}. Options that don't parse as
    "<letter><punct><text>" are skipped.
    """
    mapping = {}
    for option in options or []:
        match = _OPTION_LETTER_RE.match(option)
        if match:
            letter, text = match.group(1), match.group(2)
            text = text.strip().lower()
            if text:
                mapping[_strip_answer_punctuation(text)] = letter.lower()
    return mapping


def _marked_option_letter(prediction, options):
    """Last resort: an explicitly marked or span-final option letter, or None.

    Only fires when neither the leading-letter nor the option-TEXT path resolved the
    prediction, and only for a letter that is in range for this doc's option count.
    """
    num_options = len(options) if options else 0
    if num_options == 0:
        return None
    span = _full_answer_span(prediction)
    for pattern in _MARKED_OPTION_LETTER_RES:
        match = pattern.search(span)
        if match and ord(match.group(1)) - ord("a") < num_options:
            return match.group(1)
    return None


# A standalone number. The lookbehind refuses a digit glued to a word character, a
# dot or a hyphen, so the frame labels the manifest introduces ("Frame-1", "<frame-1>")
# cannot be read as the number -1, and "v2.5x" cannot be read as 2.5.
_NUMBER_RE = re.compile(r"(?<![\w.-])[-+]?\d*\.?\d+")

def extract_number(text, require_answer_tag=False):
    """Return the number the response ANSWERS with, or None if it did not answer.

    Deviation 6: a closed <answer></answer> span is required whenever the response
    opens one, and required outright under the think protocol
    (`require_answer_tag=True`, set per task -- see ego3d_process_results_think).
    Under the think protocol a response with no closed span is unfinished reasoning,
    so returning None makes `ego3d_process_results` apply the documented worst-case
    rule instead of crediting whatever number the reasoning happened to mention.
    Only the short protocol, which asks for "a single word or phrase" and caps
    generation at 16 tokens, is read without any tag.
    """
    span = _closed_answer_span(text)
    if span is None:
        if require_answer_tag or "<answer>" in text:
            return None
        span = text
    match = _NUMBER_RE.search(span)
    return float(match.group()) if match else None


def ego3d_process_results(doc, results, require_answer_tag=False):
    prediction = results[0]
    category = _category(doc)
    row = {
        "category": category,
        "source": doc["metadata"]["source"],
        "prediction": prediction,
        "ground_truth": doc["answer"],
    }

    if category in EXACT_NUMBER_CATEGORIES:
        target = float(doc["answer"])
        predicted = extract_number(prediction, require_answer_tag=require_answer_tag)
        if predicted is None:
            # Deviation 1: worst case rather than dropped.
            predicted = MAX_DISTANCE_M
        # Deviation 4: also lower-clips to 0, whereas upstream only upper-clips at 100.
        predicted = min(max(predicted, 0.0), MAX_DISTANCE_M)
        row["squared_error"] = (predicted - target) ** 2
    else:
        target = str(doc["answer"]).strip().lower()
        options = doc.get("options")
        # Deviation 5: the option letter may be glued to the option text with no
        # separating space ("A.1 meter", "A.ego car") rather than cleanly separated
        # ("A. yes"). Resolve a leading letter first, guarded by the doc's own
        # option count so text answers like "no"/"yes" aren't misread as letters
        # N/Y; only fall back to the bare extracted token if no valid letter leads.
        predicted = _resolve_option_letter(prediction, options)
        if predicted is None:
            predicted = extract_answer(prediction)
        if predicted != target:
            # Deviation 3: some categories' options are text-valued (e.g. "A. yes" /
            # "B. no") while the ground truth is the option LETTER. A model that
            # (correctly) answers with the option text must not be scored wrong just
            # because it didn't echo the letter. Resolve the prediction against the
            # doc's own options generically, not just for yes/no.
            option_map = _option_text_to_letter(options)
            resolved = option_map.get(predicted)
            if resolved is None:
                resolved = option_map.get(_full_answer_span(prediction))
            if resolved is None:
                # Last resort: an explicitly marked or span-final option letter,
                # e.g. "the answer is (B)." / "**B**" / "... so b".
                resolved = _marked_option_letter(prediction, options)
            if resolved is not None:
                predicted = resolved
        row["accuracy"] = float(predicted == target)

    return {"ego3d_score": row}


def ego3d_process_results_think(doc, results):
    """Scorer for the `<think>/<answer>` tasks: a numeric answer MUST be tagged.

    lmms_eval does not pass `lmms_eval_specific_kwargs` to `process_results`, so the
    protocol is selected by which of these two functions a task's yaml points at.
    """
    return ego3d_process_results(doc, results, require_answer_tag=True)


def ego3d_aggregate_results(results):
    """Log per-category scores next to their trivial floors; return mean accuracy (%)."""
    frame = pd.DataFrame(results)
    report, accuracies = {}, []

    for category, index in frame.groupby("category").groups.items():
        rows = frame.iloc[index]
        if category in EXACT_NUMBER_CATEGORIES:
            rmse = float(np.sqrt(rows["squared_error"].mean()))
            report[category] = {
                "n": len(rows),
                "rmse": round(rmse, 2),
                "constant_predictor_rmse_floor": CONSTANT_RMSE_FLOORS[category],
                "beats_floor": rmse < CONSTANT_RMSE_FLOORS[category],
            }
        else:
            accuracy = float(rows["accuracy"].mean()) * 100.0
            accuracies.append(accuracy)
            report[category] = {
                "n": len(rows),
                "accuracy": round(accuracy, 2),
                "chance_floor": CHANCE_FLOORS.get(category),
                "majority_class_floor": MAJORITY_FLOORS.get(category),
                "beats_majority": (
                    accuracy > MAJORITY_FLOORS[category]
                    if category in MAJORITY_FLOORS else None
                ),
            }

    overall = sum(accuracies) / len(accuracies) if accuracies else 0.0
    eval_logger.info("Ego3D-Bench per-category results:")
    for category in sorted(report):
        eval_logger.info(f"  {category}: {report[category]}")
    eval_logger.info(f"Ego3D-Bench mean multi-choice accuracy: {overall:.2f}")
    return overall
