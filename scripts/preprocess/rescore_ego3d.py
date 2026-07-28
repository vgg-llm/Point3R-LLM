#!/usr/bin/env python
"""Offline rescoring for Ego3D-Bench lmms_eval samples, using the CURRENT scorer.

Re-derives every metric in a saved `*_samples_*.jsonl` file with today's
`src/lmms_eval/tasks/ego3d_point3r/utils.py` scoring functions, so a fix to the
scorer can be validated without re-running (GPU) inference. No model, no GPU: this
script only reads two JSON files and calls the same pure functions lmms_eval calls.

Docs saved inside an older samples file may predate later doc-schema additions
(e.g. the `options` field). To keep results faithful to the CURRENT pipeline, this
tool re-attaches the CURRENT doc (from --docs) instead of using the doc embedded in
the samples file. `metadata.idx` alone is NOT a unique doc key in this dataset (the
converter enumerates `idx` per source+category, so e.g. `idx=1` occurs once per
(source, category) pair -- 1791 times overall out of 8675 docs). The join key used
here is therefore the full `(source, category, idx)` triple from `metadata`, which
is unique for every one of the 8675 docs.

Usage:
  python scripts/preprocess/rescore_ego3d.py \\
      --samples logs/.../..._samples_ego3d_baseline.jsonl \\
      --docs data/evaluation/ego3d_point3r/test.json
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "lmms_eval" / "tasks" / "ego3d_point3r"))

import utils as ego3d  # noqa: E402


def _unwrap(value):
    """Descend through nested single-element lists to the underlying string.

    Handles both `filtered_resps` (`["text"]`) and `resps` (`[["text"]]`) shapes.
    """
    while isinstance(value, list):
        if not value:
            raise ValueError("empty resps/filtered_resps list")
        value = value[0]
    return value


def _doc_key(metadata):
    """Unique doc identity: `metadata.idx` alone repeats across (source, category)."""
    return (metadata["source"], metadata["category"], metadata["idx"])


def load_docs_by_idx(docs_path):
    with open(docs_path) as f:
        docs = json.load(f)
    by_key = {}
    for doc in docs:
        by_key[_doc_key(doc["metadata"])] = doc
    return by_key


def load_samples(samples_path, docs_by_idx, docs_path):
    rows = []
    with open(samples_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            saved_doc = sample["doc"]
            key = _doc_key(saved_doc["metadata"])
            if key not in docs_by_idx:
                raise KeyError(
                    f"{samples_path}:{line_no}: sample with (source, category, idx) "
                    f"= {key} not found in docs file {docs_path!r}"
                )
            doc = docs_by_idx[key]

            prediction_field = sample.get("filtered_resps")
            if prediction_field is None:
                prediction_field = sample.get("resps")
            if prediction_field is None:
                raise KeyError(
                    f"{samples_path}:{line_no}: sample idx {idx} has neither "
                    "filtered_resps nor resps"
                )
            prediction = _unwrap(prediction_field)

            rows.append(ego3d.ego3d_process_results(doc, [prediction])["ego3d_score"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="lmms_eval samples JSONL")
    parser.add_argument(
        "--docs", required=True,
        help="current task docs (e.g. data/evaluation/ego3d_point3r/test.json)",
    )
    args = parser.parse_args()

    docs_by_idx = load_docs_by_idx(args.docs)
    rows = load_samples(args.samples, docs_by_idx, args.docs)
    print(f"Rescored {len(rows)} samples against current utils.py scoring logic.\n")
    overall = ego3d.ego3d_aggregate_results(rows)
    print(f"\nEgo3D-Bench mean multi-choice accuracy: {overall:.2f}")


if __name__ == "__main__":
    main()
