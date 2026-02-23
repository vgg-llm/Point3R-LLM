"""
SPAR-Subset Point3R Evaluation Script

Evaluates appearance ordering tasks from SPAR-subset.
Question types: fill (concise comma-separated), sentence (verbose natural language)
Data sources: scannet, scannetpp
"""

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

with open(Path(__file__).parent / "spar_subset_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]


def spar_subset_doc_to_visual(doc):
    """Return empty list since Point3R uses pointer_data instead of images/videos."""
    return [[]]


def spar_subset_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract question text from conversations."""
    question = doc["conversations"][0]["value"]
    return question


def exact_match(pred, target):
    """Check if prediction matches target (case-insensitive)."""
    return 1.0 if pred.strip().lower() == target.strip().lower() else 0.0


def spar_subset_process_results(doc, results):
    """Process results for a single document."""
    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(doc["prediction"], doc["ground_truth"])
    return {"spar_subset_point3r_score": doc}


def spar_subset_aggregate_results(results):
    """Aggregate results across all documents."""
    results = pd.DataFrame(results)

    output = {}

    # Group by question_type (fill/sentence)
    for question_type, qt_indexes in results.groupby("question_type").groups.items():
        per_qt = results.iloc[qt_indexes]
        output[f"{question_type}_accuracy"] = per_qt["accuracy"].mean()

        # Further group by data_source within each question_type
        for data_source, ds_indexes in per_qt.groupby("data_source").groups.items():
            per_ds = per_qt.iloc[ds_indexes]
            output[f"{data_source}_{question_type}_accuracy"] = per_ds["accuracy"].mean()

    # Compute overall score
    if output:
        # Average across question_type-level scores only (not the per-source breakdowns)
        qt_scores = [v for k, v in output.items() if k.count("_") == 1]
        output["overall"] = sum(qt_scores) / len(qt_scores) if qt_scores else 0.0
    else:
        output["overall"] = 0.0

    eval_logger.info(f"SPAR-Subset Point3R Evaluation results: {output}")
    return output["overall"] * 100.0
