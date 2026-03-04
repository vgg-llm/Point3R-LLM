#!/usr/bin/env python3
"""
Multi-GPU re-scoring for RoboFAC evaluation.

Shards open-ended LLM scoring across multiple GPUs using multiprocessing,
then runs the full aggregation.

Usage:
  # 8 GPUs, default model:
  python scripts/demo/rescore_robofac_multi.py \
    --jsonl_path <path_to_jsonl> --num_gpus 8

  # Custom model and output:
  python scripts/demo/rescore_robofac_multi.py \
    --jsonl_path <path_to_jsonl> --num_gpus 8 \
    --model_path Qwen/Qwen3-VL-4B-Instruct --output results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

# Add src/ to path (no heavy imports at module level -- workers need
# CUDA_VISIBLE_DEVICES set before torch is imported)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Open-ended question types that need LLM scoring
OPEN_ENDED_QUESTION_TYPES = {
    "Task identification",
    "Task planning",
    "Failure explanation",
    "High-level correction",
    "Low-level correction",
}


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


def _make_eval_prompt(question, pred, ref):
    """Build LLM evaluation prompt (matching utils.py)."""
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


def _score_from_result(result):
    """Extract score (0-100) from parsed LLM JSON result."""
    if result is None:
        return 0.0
    try:
        c = result["criteria"]
        return (c["correctness"]["score"] + c["relevance"]["score"] + c["completeness"]["score"]) / 3 * 20
    except (KeyError, TypeError):
        return 0.0


def worker_fn(rank, prompts_shard, model_path, output_file):
    """Load model on assigned GPU, score prompts, write results to file."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"[GPU {rank}] Loading model {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir="./cache", trust_remote_code=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir="./cache",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[GPU {rank}] Model loaded. Scoring {len(prompts_shard)} prompts...")

    results = []
    for i, prompt in enumerate(prompts_shard):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        content = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = _parse_llm_json(content)
        if parsed is None:
            print(f"[GPU {rank}] Parse failed for prompt {i}: {content[:200]}")
        results.append(parsed)

        if (i + 1) % 50 == 0:
            print(f"[GPU {rank}] Progress: {i + 1}/{len(prompts_shard)}")

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"[GPU {rank}] Done.")

    del model, tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU RoboFAC rescoring")
    parser.add_argument("--jsonl_path", required=True,
                        help="Path to samples JSONL from lmms_eval")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="HF model ID or local path for LLM judge")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for detailed results")
    parser.add_argument("--task", choices=["robofac", "robofac_point3r"],
                        default="robofac_point3r",
                        help="Task variant for aggregation")
    args = parser.parse_args()

    import torch
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()

    # --- Read JSONL ---
    docs = []
    with open(args.jsonl_path) as f:
        for line in f:
            docs.append(json.loads(line.strip())["doc"])
    print(f"Loaded {len(docs)} samples from {args.jsonl_path}")

    # --- Build prompts for open-ended questions ---
    oe_indices = []
    prompts = []
    for i, doc in enumerate(docs):
        if doc["question_type"] in OPEN_ENDED_QUESTION_TYPES:
            oe_indices.append(i)
            prompts.append(_make_eval_prompt(
                doc["question"], doc["prediction"], doc["ground_truth"]
            ))

    print(f"Open-ended questions: {len(prompts)}, using {args.num_gpus} GPUs")

    # --- Shard prompts across GPUs (contiguous) ---
    num_gpus = args.num_gpus
    shard_size = len(prompts) // num_gpus
    shards = []
    for rank in range(num_gpus):
        start = rank * shard_size
        end = (start + shard_size) if rank < num_gpus - 1 else len(prompts)
        shards.append(prompts[start:end])

    # --- Spawn workers ---
    tmp_dir = tempfile.mkdtemp(prefix="rescore_")
    output_files = [os.path.join(tmp_dir, f"shard_{rank}.json") for rank in range(num_gpus)]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(rank, shards[rank], args.model_path, output_files[rank]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: Worker PID {p.pid} exited with code {p.exitcode}")

    # --- Gather results ---
    all_llm_results = []
    for rank in range(num_gpus):
        with open(output_files[rank]) as f:
            all_llm_results.extend(json.load(f))
        os.remove(output_files[rank])
    os.rmdir(tmp_dir)

    assert len(all_llm_results) == len(oe_indices), \
        f"Result count mismatch: {len(all_llm_results)} vs {len(oe_indices)}"

    # --- Set llm_score on each open-ended doc ---
    scored = 0
    for idx, result in zip(oe_indices, all_llm_results):
        docs[idx]["llm_score"] = _score_from_result(result)
        if result is not None:
            scored += 1
    print(f"LLM scoring complete: {scored}/{len(oe_indices)} successfully parsed")

    # --- Run aggregation (skips LLM re-scoring since llm_score is pre-set) ---
    if args.task == "robofac_point3r":
        from lmms_eval.tasks.robofac_point3r.utils import robofac_aggregate_results
    else:
        from lmms_eval.tasks.robofac.utils import robofac_aggregate_results

    overall = robofac_aggregate_results(docs)
    print(f"\nOverall score: {overall:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"overall": overall}, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
