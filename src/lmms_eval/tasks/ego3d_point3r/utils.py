"""Ego3D-Bench evaluation for Point3R-LLM.

Ports Ego3D-Bench/utils/eval.py: lowercased exact match for the 8 multi-choice
categories, RMSE (predictions clipped to 100 m) for the 2 absolute-distance ones.

Two deliberate deviations from upstream, both strictly harsher:
  1. Unparseable numeric predictions score worst-case instead of being dropped
     from the RMSE. Upstream's `if pred:` also drops legitimate 0 predictions.
  2. No resume logic; lmms_eval owns sharding and logging.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

with open(Path(__file__).parent / "ego3d_point3r.yaml", "r") as f:
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


def extract_answer(text):
    """Return the comparable multi-choice answer, lowercased.

    Protocol-agnostic: reads the <answer> span when present, else the first token.
    """
    if "<answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        text = text[start:end] if end != -1 else text[start:]
    token = text.replace("\n", " ").strip().split(" ")[0]
    return token.rstrip(".").strip().lower()


def extract_number(text):
    """Return the first number in the text, or None."""
    if "<answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        text = text[start:end] if end != -1 else text[start:]
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else None


def ego3d_process_results(doc, results):
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
        predicted = extract_number(prediction)
        if predicted is None:
            # Deviation 1: worst case rather than dropped.
            predicted = MAX_DISTANCE_M
        predicted = min(max(predicted, 0.0), MAX_DISTANCE_M)
        row["squared_error"] = (predicted - target) ** 2
    else:
        row["accuracy"] = float(
            extract_answer(prediction) == str(doc["answer"]).strip().lower()
        )

    return {"ego3d_score": row}


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
