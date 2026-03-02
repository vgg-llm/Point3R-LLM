"""
RoboFAC Point3R Evaluation Script (Pointer memory-based)

Evaluates robot failure analysis and correction using pointer memory features.

Question types:
- Choice-based (string match): Failure detection, Failure identification, Failure locating
- Open-ended (LLM scoring): Task identification, Task planning, Failure explanation,
                              High-level correction, Low-level correction

Environment variables for LLM-based scoring (pick one):
  Option 1 - OpenAI API:
    OPENAI_API_KEY: API key
    ROBOFAC_LLM_MODEL: model name (default: "gpt-4o")
    ROBOFAC_LLM_URL: API base URL (default: "https://api.openai.com/v1")
  Option 2 - Local model via transformers:
    ROBOFAC_LLM_LOCAL: path to local model (e.g. "cache/models--Qwen--Qwen3-VL-4B-Instruct")
  Option 3 - Local vLLM server:
    ROBOFAC_LLM_URL: e.g. "http://localhost:8000/v1"
    ROBOFAC_LLM_MODEL: model name served by vLLM
    OPENAI_API_KEY: set to any non-empty string (e.g. "local")

If none are set, falls back to contains_match scoring.
"""

import json
import os
from pathlib import Path

import torch
import pandas as pd
import yaml
from loguru import logger as eval_logger

with open(Path(__file__).parent / "robofac_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

config = yaml.safe_load("".join(safe_data))
dataset_path = config["dataset_path"]

# Choice-based question types (scored by string matching)
CHOICE_QUESTION_TYPES = [
    "Failure detection",
    "Failure identification",
    "Failure locating",
]

# Open-ended question types (scored by GPT-4O or fallback)
OPEN_ENDED_QUESTION_TYPES = [
    "Task identification",
    "Task planning",
    "Failure explanation",
    "High-level correction",
    "Low-level correction",
]

# Format hints for non-RoboFAC models (matching original eval script)
FORMAT_HINTS = {
    "High-level correction": " (Your answer should be a detailed textual suggestion of about two to three sentences.)",
    "Low-level correction": " (Your answer should describe which direction (relative to the robot arm) and how much the robot arm should move to recover from the failure.)",
    "Failure explanation": " (Your answer should be a detailed textual description of about two to three sentences.)",
    "Task planning": " (Your answer should be a series of steps in the format of '1. ... 2. ... ...'.)",
    "Task identification": " (Your answer should be a brief phrase that describes the task in the video.)",
}

# Experiment-type partitioning (per RoboFAC paper)
EXPERIMENT_TYPES = {
    "Short-horizon": ["PickCube", "PullCube", "PushCube", "StackCube", "LiftPegUpright"],
    "Medium-horizon": ["PlugCharger", "PegInsertionSide", "UprightStack", "PullCubeTool",
                        "InsertCylinder", "PlaceCube", "ToolsTask"],
    "Long-horizon": ["SafeTask", "MicrowaveTask"],
    "Dynamic": ["SpinStack", "SpinPullStack"],
}
TASK_TO_EXPERIMENT_TYPE = {
    task: exp_type
    for exp_type, tasks in EXPERIMENT_TYPES.items()
    for task in tasks
}

MEDIA_BASE_DIR = "data/media/robofac"


def robofac_doc_to_visual(doc):
    """Load video from video_path."""
    video_path = doc.get("video_path", "")
    full_path = os.path.join(MEDIA_BASE_DIR, video_path)

    if not os.path.exists(full_path):
        eval_logger.warning(f"Video not found: {full_path}")

    return [full_path]


def robofac_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format question with appropriate prompt."""
    question = doc["question"]
    question_type = doc.get("question_type", "")

    pre_prompt = "These are frames of a video showing a robotic arm performing a task."
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt) or pre_prompt

    # Add format hints for open-ended questions
    hint = FORMAT_HINTS.get(question_type, "")

    if question_type in CHOICE_QUESTION_TYPES:
        options = doc.get("options", None)
        post_prompt = ""
        if lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs.get("choice_post_prompt", "") or ""

        if options and question_type == "Failure detection":
            options_text = "Options: Yes / No"
        elif options:
            options_text = "Options:\n" + "\n".join(
                f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
            )
        else:
            options_text = ""

        parts = [pre_prompt, question + hint]
        if options_text:
            parts.append(options_text)
        if post_prompt:
            parts.append(post_prompt)
        return "\n".join(parts)
    else:
        post_prompt = ""
        if lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs.get("open_post_prompt", "") or ""

        parts = [pre_prompt, question + hint]
        if post_prompt:
            parts.append(post_prompt)
        return "\n".join(parts)


def robofac_process_results(doc, results):
    """Process results for a single document. Scoring is deferred to aggregate."""
    doc["prediction"] = results[0]
    return {"robofac_score": doc}


def _make_eval_prompt(question, pred, ref):
    """Build GPT-4O evaluation prompt (from original RoboFAC eval_utils.py)."""
    return f"""You are an expert evaluator. Assess the quality of a model's response to the user's query.

        Question: {question}

        Reference answer: {ref}

        Model's response: {pred}

        Evaluate the model's response on the following criteria:
        - correctness: factual accuracy and consistency with the reference answer.
        - relevance: how well the model's response addresses the question.
        - completeness: whether all key aspects of the reference answer are covered.

        For each criterion, provide a score from 0 to 5 and a **brief** explanation, the score should be an integer.
        The score you give needs to be strict and demanding.

        Output ONLY the JSON object in the following format:
        {{
        "criteria": {{
            "correctness": {{"score": <0-5>, "explanation": <brief explanation>}},
            "relevance": {{"score": <0-5>, "explanation": <brief explanation>}},
            "completeness": {{"score": <0-5>, "explanation": <brief explanation>}},
        }}
        }}
        """


def _parse_llm_json(content):
    """Parse JSON from LLM response, with fallback regex extraction."""
    import re
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        json_match = re.search(r"\{[\s\S]*\}", content or "")
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    return None


def _call_llm_batch_api(prompts, model_name, base_url, api_key):
    """Call OpenAI-compatible API for a batch of prompts with async concurrency."""
    import asyncio
    import httpx

    CONCURRENCY = 30  # matching original RoboFAC semaphore size

    async def _single_request(sem, client, idx, prompt):
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                        },
                        timeout=httpx.Timeout(connect=10, read=200, write=30, pool=10),
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    parsed = _parse_llm_json(content)
                    if parsed is not None:
                        return idx, parsed
                    if attempt == 2:
                        eval_logger.warning(f"Failed to parse LLM response for prompt {idx}: {content[:200]}")
                        return idx, None
                except Exception as e:
                    if attempt == 2:
                        eval_logger.warning(f"LLM API error for prompt {idx}: {e}")
                        return idx, None

    async def _run_all():
        sem = asyncio.Semaphore(CONCURRENCY)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(headers=headers) as client:
            tasks = [_single_request(sem, client, i, p) for i, p in enumerate(prompts)]
            completed = 0
            results_map = {}
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results_map[idx] = result
                completed += 1
                if completed % 50 == 0:
                    eval_logger.info(f"LLM evaluation progress: {completed}/{len(prompts)}")
            return [results_map[i] for i in range(len(prompts))]

    return asyncio.run(_run_all())


def _resolve_model_path(path):
    """Resolve HuggingFace cache directory to actual model snapshot path."""
    p = Path(path)
    snapshots = p / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted(
            [d for d in snapshots.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if snapshot_dirs:
            resolved = str(snapshot_dirs[0])
            eval_logger.info(f"Resolved HF cache path: {path} -> {resolved}")
            return resolved
    return path


def _call_llm_batch_local(prompts, model_path):
    """Call a local transformers model for scoring."""
    from transformers import AutoTokenizer

    model_path = _resolve_model_path(model_path)
    eval_logger.info(f"Loading local model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Try CausalLM first, fall back to Vision2Seq for VL models
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except (ValueError, KeyError):
        from transformers import AutoModelForVision2Seq
        eval_logger.info("CausalLM loading failed, trying Vision2Seq (VL model)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    results = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 50 == 0:
            eval_logger.info(f"Local LLM evaluation progress: {i + 1}/{len(prompts)}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        # Decode only newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        content = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = _parse_llm_json(content)
        if parsed is None:
            eval_logger.warning(f"Failed to parse local LLM response for prompt {i}: {content[:200]}")
        results.append(parsed)

    # Free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def _score_open_ended_with_llm(open_ended_results):
    """Score open-ended questions using LLM (API or local)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model_name = os.environ.get("ROBOFAC_LLM_MODEL", "gpt-4o")
    base_url = os.environ.get("ROBOFAC_LLM_URL", "https://api.openai.com/v1")
    local_model = os.environ.get("ROBOFAC_LLM_LOCAL", "")

    use_api = bool(api_key)
    use_local = bool(local_model)

    if not use_api and not use_local:
        eval_logger.warning(
            "No LLM configured for open-ended scoring. "
            "Set OPENAI_API_KEY (API) or ROBOFAC_LLM_LOCAL (local model). "
            "Falling back to contains_match."
        )
        for doc in open_ended_results:
            pred = doc["prediction"].strip()
            target = doc["ground_truth"].strip()
            doc["llm_score"] = 100.0 if target.lower() in pred.lower() else 0.0
        return

    prompts = [
        _make_eval_prompt(doc["question"], doc["prediction"], doc["ground_truth"])
        for doc in open_ended_results
    ]

    if use_local:
        eval_logger.info(
            f"Scoring {len(open_ended_results)} open-ended questions with local model: {local_model}"
        )
        llm_results = _call_llm_batch_local(prompts, local_model)
    else:
        eval_logger.info(
            f"Scoring {len(open_ended_results)} open-ended questions with {model_name} ({base_url})"
        )
        llm_results = _call_llm_batch_api(prompts, model_name, base_url, api_key)

    for doc, result in zip(open_ended_results, llm_results):
        if result is not None:
            try:
                criteria = result["criteria"]
                correctness = criteria["correctness"]["score"]
                relevance = criteria["relevance"]["score"]
                completeness = criteria["completeness"]["score"]
                # Overall: (sum / 3) * 20, matching original RoboFAC formula (max 100)
                doc["llm_score"] = (correctness + relevance + completeness) / 3 * 20
            except (KeyError, TypeError):
                eval_logger.warning(f"Invalid LLM result format: {result}")
                doc["llm_score"] = 0.0
        else:
            doc["llm_score"] = 0.0


def _get_experiment_type(row):
    """Determine experiment type from data_source and task name."""
    if row.get("data_source") == "realworld":
        return "Real-world"
    task = row.get("task", "")
    return TASK_TO_EXPERIMENT_TYPE.get(task, "Unknown")


def _score_choice_subset(df):
    """Score choice-based questions in a DataFrame subset. Returns dict of qt -> accuracy (0-100)."""
    scores = {}
    for qt in CHOICE_QUESTION_TYPES:
        qt_mask = df["question_type"] == qt
        if not qt_mask.any():
            continue
        qt_docs = df[qt_mask]
        correct = 0
        total = 0
        for _, doc in qt_docs.iterrows():
            pred = doc["prediction"].strip().lower()
            target = doc["ground_truth"].strip().lower()
            if pred in target:
                correct += 1
            total += 1
        scores[qt] = (correct / total * 100) if total > 0 else 0.0
    return scores


def _score_open_ended_subset(docs_list):
    """Score open-ended questions in a list of doc dicts. Returns dict of qt -> score (0-100)."""
    from collections import defaultdict
    qt_scores = defaultdict(list)
    for doc in docs_list:
        qt_scores[doc["question_type"]].append(doc["llm_score"])
    return {qt: sum(s) / len(s) for qt, s in qt_scores.items()}


def robofac_aggregate_results(results):
    """Aggregate results with experiment-type partitioning (per RoboFAC paper)."""
    results_df = pd.DataFrame(results)

    # Assign experiment type to each row
    results_df["experiment_type"] = results_df.apply(_get_experiment_type, axis=1)

    output = {}

    # === Score all open-ended questions first (single LLM batch) ===
    open_ended_docs = []
    for _, row in results_df.iterrows():
        if row["question_type"] in OPEN_ENDED_QUESTION_TYPES:
            open_ended_docs.append(row.to_dict())

    if open_ended_docs:
        _score_open_ended_with_llm(open_ended_docs)
        # Write llm_score back into results_df
        oe_idx = 0
        for i, row in results_df.iterrows():
            if row["question_type"] in OPEN_ENDED_QUESTION_TYPES:
                results_df.at[i, "llm_score"] = open_ended_docs[oe_idx]["llm_score"]
                oe_idx += 1

    # === Per-experiment-type scoring ===
    exp_type_order = ["Short-horizon", "Medium-horizon", "Long-horizon", "Dynamic", "Real-world"]
    exp_type_scores = {}  # exp_type -> (score, sample_count)

    for exp_type in exp_type_order:
        exp_mask = results_df["experiment_type"] == exp_type
        if not exp_mask.any():
            continue
        exp_df = results_df[exp_mask]
        n_samples = len(exp_df)

        # Choice-based scores
        choice_scores = _score_choice_subset(exp_df)

        # Open-ended scores
        oe_docs = [row.to_dict() for _, row in exp_df.iterrows()
                    if row["question_type"] in OPEN_ENDED_QUESTION_TYPES and "llm_score" in row]
        open_scores = _score_open_ended_subset(oe_docs) if oe_docs else {}

        # Combine all question-type scores for this experiment type
        qt_scores = []
        for qt in CHOICE_QUESTION_TYPES:
            if qt in choice_scores:
                qt_scores.append(choice_scores[qt])
                output[f"{exp_type}/{qt}_accuracy"] = choice_scores[qt]
        for qt in OPEN_ENDED_QUESTION_TYPES:
            if qt in open_scores:
                qt_scores.append(open_scores[qt])
                output[f"{exp_type}/{qt}_score"] = open_scores[qt]

        exp_avg = sum(qt_scores) / len(qt_scores) if qt_scores else 0.0
        output[f"{exp_type}_score"] = exp_avg
        exp_type_scores[exp_type] = (exp_avg, n_samples)

        eval_logger.info(f"  {exp_type}: {exp_avg:.1f}/100 ({n_samples} samples)")
        for qt in CHOICE_QUESTION_TYPES:
            if qt in choice_scores:
                eval_logger.info(f"    {qt}: {choice_scores[qt]:.1f}%")
        for qt in OPEN_ENDED_QUESTION_TYPES:
            if qt in open_scores:
                eval_logger.info(f"    {qt}: {open_scores[qt]:.1f}/100")

    # === Global per-question-type scores (for backward compat) ===
    global_choice = _score_choice_subset(results_df)
    for qt, score in global_choice.items():
        output[f"{qt}_accuracy"] = score

    if open_ended_docs:
        from collections import defaultdict
        qt_scores_global = defaultdict(list)
        for doc in open_ended_docs:
            qt_scores_global[doc["question_type"]].append(doc["llm_score"])
        for qt, scores in qt_scores_global.items():
            output[f"{qt}_score"] = sum(scores) / len(scores)

    # === Overall score (sample-weighted average across experiment types) ===
    total_weight = sum(n for _, n in exp_type_scores.values())
    if total_weight > 0:
        overall = sum(score * n for score, n in exp_type_scores.values()) / total_weight
    else:
        overall = 0.0
    output["overall"] = overall

    # Warn about unknown experiment types
    unknown_mask = results_df["experiment_type"] == "Unknown"
    if unknown_mask.any():
        unknown_tasks = results_df[unknown_mask]["task"].unique().tolist()
        eval_logger.warning(f"Unknown experiment type for tasks: {unknown_tasks}")

    eval_logger.info(f"RoboFAC Point3R Overall: {overall:.1f}/100")
    eval_logger.info(f"RoboFAC Point3R Full results: {output}")
    return overall
