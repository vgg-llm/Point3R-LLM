"""
Prepare RoboFAC evaluation data for lmms_eval.

Flattens the 40 per-video split files from data/media/robofac/test_qa_sim/
into a single data/robofac/test.json with one entry per QA pair.

Usage:
    python scripts/data/prepare_robofac_eval.py
"""

import json
import os
import re
from collections import Counter

BASE_DIR = "data/media/robofac"
TEST_DIR = os.path.join(BASE_DIR, "test_qa_sim")
OUTPUT_DIR = "data/robofac"


def clean_question(text):
    """Remove <image> tokens from question text.

    The original test data has '<image>\\n' prefix in questions. In lmms_eval,
    visual input is provided separately via doc_to_visual(), so the model wrapper
    handles composing visuals + text with its own tokens.
    """
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def extract_options(question_text, question_type):
    """Extract options from question text for choice-based questions."""
    if question_type == "Failure detection":
        return ["Yes", "No"]

    # For Failure identification and Failure locating, options are in brackets like:
    # (Your answer should choose one of the following options: ['Option1.', 'Option2.', ...])
    match = re.search(r"\[(['\"].*?['\"](?:\s*,\s*['\"].*?['\"])*)\]", question_text)
    if match:
        raw = match.group(1)
        options = re.findall(r"['\"]([^'\"]+)['\"]", raw)
        return options
    return None


CHOICE_QUESTION_TYPES = {
    "Failure detection",
    "Failure identification",
    "Failure locating",
}


def main():
    entries = []
    idx = 0
    qt_counts = Counter()

    for split_i in range(40):
        filepath = os.path.join(TEST_DIR, f"annos_per_video_split{split_i}.json")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for video_id, video_entry in data.items():
            video_path = os.path.join("simulation_data", video_entry["video"])
            task_name = video_entry["task"]

            for question_type, convos in video_entry["annos"].items():
                # Conversations are [human_msg, assistant_msg] pairs
                for i in range(0, len(convos), 2):
                    if i + 1 >= len(convos):
                        break

                    human_msg = convos[i]
                    assistant_msg = convos[i + 1]

                    assert human_msg["from"] == "human", f"Expected 'human', got '{human_msg['from']}'"
                    assert assistant_msg["from"] == "assistant", f"Expected 'assistant', got '{assistant_msg['from']}'"

                    question = clean_question(human_msg["value"])
                    answer = assistant_msg["value"]

                    options = None
                    if question_type in CHOICE_QUESTION_TYPES:
                        options = extract_options(human_msg["value"], question_type)

                    entry = {
                        "id": idx,
                        "video_path": video_path,
                        "question": question,
                        "question_type": question_type,
                        "ground_truth": answer,
                        "task": task_name,
                        "options": options,
                    }
                    entries.append(entry)
                    qt_counts[question_type] += 1
                    idx += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "test.json")
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Total entries: {len(entries)}")
    print(f"Output: {output_path}")
    print(f"\nQuestion type distribution:")
    for qt, count in sorted(qt_counts.items()):
        print(f"  {qt}: {count}")

    # Validate options extraction for choice-based questions
    for qt in CHOICE_QUESTION_TYPES:
        qt_entries = [e for e in entries if e["question_type"] == qt]
        with_options = sum(1 for e in qt_entries if e["options"])
        without_options = sum(1 for e in qt_entries if not e["options"])
        if without_options > 0:
            print(f"\nWarning: {qt} has {without_options}/{len(qt_entries)} entries without extracted options")


if __name__ == "__main__":
    main()
