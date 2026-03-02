"""
Prepare RoboFAC evaluation data for lmms_eval.

Flattens the per-video split files from data/media/robofac/test_qa_sim/ and
test_qa_realworld/ into a single data/robofac/test.json with one entry per QA pair.

Each entry includes a `data_source` field ("sim" or "realworld") for experiment-type
partitioning during evaluation.

Usage:
    python scripts/data/prepare_robofac_eval.py
"""

import json
import os
import re
from collections import Counter

BASE_DIR = "data/media/robofac"
SIM_TEST_DIR = os.path.join(BASE_DIR, "test_qa_sim")
REALWORLD_TEST_DIR = os.path.join(BASE_DIR, "test_qa_realworld")
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


def fix_legacy_names(video_path, task_name):
    """Convert old dataset naming to current naming convention (per RoboFAC GitHub issue)."""
    parts = video_path.split("/")
    parts[0] = parts[0].replace('dataset_success_cleaned', 'success_data')
    parts[0] = parts[0].replace('Rotation-v1', 'SpinStack-v1')
    parts[0] = parts[0].replace('Rotation-gen1', 'SpinStack-gen1')
    parts[0] = parts[0].replace('Rotation-gen2', 'SpinStack-gen2')
    parts[0] = parts[0].replace('Rotation2-v1', 'SpinPullStack-v1')
    parts[0] = parts[0].replace('Rotation2-gen1', 'SpinPullStack-gen1')
    parts[0] = parts[0].replace('Rotation2-gen2', 'SpinPullStack-gen2')
    video_path = "/".join(parts)
    # Order matters: replace Rotation2 before Rotation to avoid partial match
    task_name = task_name.replace('Rotation2', 'SpinPullStack').replace('Rotation', 'SpinStack')
    return video_path, task_name


def process_splits(test_dir, video_base_dir, data_source, max_splits=40):
    """Process annotation split files from a test directory.

    Returns list of entries and question-type counts.
    """
    entries = []
    qt_counts = Counter()

    for split_i in range(max_splits):
        filepath = os.path.join(test_dir, f"annos_per_video_split{split_i}.json")
        if not os.path.exists(filepath):
            if split_i == 0:
                print(f"Warning: {filepath} not found, skipping {data_source} data")
            break

        with open(filepath) as f:
            data = json.load(f)

        for video_id, video_entry in data.items():
            video_path = os.path.join(video_base_dir, video_entry["video"])
            task_name = video_entry["task"]

            # Fix legacy names for simulation data
            if data_source == "sim":
                video_path, task_name = fix_legacy_names(video_path, task_name)

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
                        "video_path": video_path,
                        "question": question,
                        "question_type": question_type,
                        "ground_truth": answer,
                        "task": task_name,
                        "data_source": data_source,
                        "options": options,
                    }
                    entries.append(entry)
                    qt_counts[question_type] += 1

    return entries, qt_counts


def main():
    # Process simulation data
    print("Processing simulation data...")
    sim_entries, sim_qt = process_splits(SIM_TEST_DIR, "simulation_data", "sim", max_splits=40)
    print(f"  Simulation entries: {len(sim_entries)}")

    # Process real-world data
    print("Processing real-world data...")
    rw_entries, rw_qt = process_splits(REALWORLD_TEST_DIR, "realworld_data", "realworld", max_splits=40)
    print(f"  Real-world entries: {len(rw_entries)}")

    # Combine and assign sequential IDs
    all_entries = sim_entries + rw_entries
    for idx, entry in enumerate(all_entries):
        entry["id"] = idx

    qt_counts = sim_qt + rw_qt

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "test.json")
    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nTotal entries: {len(all_entries)}")
    print(f"Output: {output_path}")

    # Task distribution
    task_counts = Counter(e["task"] for e in all_entries)
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        source = "sim" if any(e["task"] == task and e["data_source"] == "sim" for e in all_entries) else "realworld"
        print(f"  {task}: {count} ({source})")

    print(f"\nQuestion type distribution:")
    for qt, count in sorted(qt_counts.items()):
        print(f"  {qt}: {count}")

    # Validate options extraction for choice-based questions
    for qt in CHOICE_QUESTION_TYPES:
        qt_entries = [e for e in all_entries if e["question_type"] == qt]
        without_options = sum(1 for e in qt_entries if not e["options"])
        if without_options > 0:
            print(f"\nWarning: {qt} has {without_options}/{len(qt_entries)} entries without extracted options")


if __name__ == "__main__":
    main()
