#!/usr/bin/env python3
"""
Re-score RoboFAC Point3R evaluation from saved JSONL results.

Reads a JSONL file produced by lmms_eval (containing doc dicts with predictions),
and re-runs ONLY the scoring/aggregation step -- no model inference.

Environment variables for open-ended scoring (same as the main eval):
  ROBOFAC_LLM_LOCAL: path to local model (e.g., "Qwen/Qwen3-VL-4B-Instruct")
  OPENAI_API_KEY + ROBOFAC_LLM_MODEL: API-based scoring
  Neither: falls back to contains_match

Usage:
  # Reproduce original score (contains_match fallback):
  python scripts/demo/rescore_robofac.py --jsonl_path <path_to_jsonl>

  # Rescore with local Qwen3-VL model:
  ROBOFAC_LLM_LOCAL="Qwen/Qwen3-VL-4B-Instruct" python scripts/demo/rescore_robofac.py --jsonl_path <path_to_jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

# Add src/ to PYTHONPATH for importing lmms_eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lmms_eval.tasks.robofac_point3r.utils import robofac_aggregate_results


def main():
    parser = argparse.ArgumentParser(description="Re-score RoboFAC from saved JSONL")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to the samples JSONL file from lmms_eval")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write results JSON")
    args = parser.parse_args()

    # Read all doc dicts from the JSONL
    docs = []
    with open(args.jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            doc = entry["doc"]
            assert "prediction" in doc, f"Line {line_num}: missing 'prediction' in doc"
            assert "ground_truth" in doc, f"Line {line_num}: missing 'ground_truth' in doc"
            assert "question_type" in doc, f"Line {line_num}: missing 'question_type' in doc"
            docs.append(doc)

    print(f"Loaded {len(docs)} samples from {args.jsonl_path}")

    # Run aggregation (calls _score_open_ended_with_llm internally)
    overall_score = robofac_aggregate_results(docs)

    print(f"\nOverall score: {overall_score:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"overall": overall_score}, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
