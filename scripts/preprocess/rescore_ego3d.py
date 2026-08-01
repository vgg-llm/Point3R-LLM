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


_THINK_SUFFIXES = (
    ego3d.THINK_SUFFIX_NUMBER,
    ego3d.THINK_SUFFIX_YESNO,
    ego3d.THINK_SUFFIX_LETTER,
)
_SHORT_SUFFIXES = (ego3d.SHORT_SUFFIX_NUMBER, ego3d.SHORT_SUFFIX_LETTER)


def _requires_answer_tag(sample, samples_path, line_no):
    """Read the protocol back off the saved prompt.

    The scorer treats numeric answers differently under the two protocols (a think
    response with no closed <answer> span is unfinished reasoning, not an answer), and
    lmms_eval selects that per task via `ego3d_process_results_think`. A samples file
    records the rendered prompt in `input`, whose suffix names the protocol exactly, so
    rescoring does not have to be told which run it is looking at.
    """
    prompt = (sample.get("input") or "").rstrip()
    if prompt.endswith(tuple(s.strip() for s in _THINK_SUFFIXES)):
        return True
    if prompt.endswith(tuple(s.strip() for s in _SHORT_SUFFIXES)):
        return False
    raise ValueError(
        f"{samples_path}:{line_no}: cannot tell the protocol from the saved prompt; "
        "its suffix matches neither the think nor the short protocol"
    )


def _doc_key(metadata):
    """Unique doc identity: `metadata.idx` alone repeats across (source, category)."""
    return (metadata["source"], metadata["category"], metadata["idx"])


def load_docs_by_idx(docs_path):
    with open(docs_path) as f:
        docs = json.load(f)
    by_key = {}
    for doc in docs:
        key = _doc_key(doc["metadata"])
        # The (source, category, idx) triple is the join key; if it ever stops being
        # unique, docs would silently shadow each other and samples would be rescored
        # against the wrong doc (wrong options -> wrong accuracy).
        assert key not in by_key, f"{docs_path}: duplicate doc key {key}"
        by_key[key] = doc
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
                    f"{samples_path}:{line_no}: sample with (source, category, idx) "
                    f"= {key} has neither filtered_resps nor resps"
                )
            prediction = _unwrap(prediction_field)

            require_answer_tag = _requires_answer_tag(sample, samples_path, line_no)
            rows.append(
                ego3d.ego3d_process_results(
                    doc, [prediction], require_answer_tag=require_answer_tag
                )["ego3d_score"]
            )
    return rows


EXPECTED_CATEGORIES = 10


def warn_if_partial(rows, num_docs):
    """Warn loudly when the rescored rows are not the full benchmark.

    The printed mean looks like a benchmark number whatever subset it came from (a
    --limit run, a crashed run, one shard). Anything less than every doc in every
    category is NOT comparable to a reported Ego3D-Bench score.
    """
    categories = {row["category"] for row in rows}
    warnings = []
    if len(rows) != num_docs:
        warnings.append(
            f"rescored {len(rows)} samples but the docs file holds {num_docs}: "
            "this is a PARTIAL run (--limit, a crash, or a single shard)"
        )
    if len(categories) < EXPECTED_CATEGORIES:
        warnings.append(
            f"only {len(categories)} of {EXPECTED_CATEGORIES} categories present: "
            f"missing "
            f"{sorted((set(ego3d.CHANCE_FLOORS) | set(ego3d.CONSTANT_RMSE_FLOORS)) - categories)}"
        )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if warnings:
        print(
            "WARNING: the mean below is NOT a benchmark number; it covers only the "
            "rows present in this samples file.",
            file=sys.stderr,
        )


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
    warn_if_partial(rows, len(docs_by_idx))


if __name__ == "__main__":
    main()
