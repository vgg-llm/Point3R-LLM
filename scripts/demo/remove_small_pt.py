#!/usr/bin/env python3
"""
Find and remove broken .pt files in a pointer_memory folder.

Three subcommands:
  1. small  — fast: remove files below a size threshold (no torch.load)
  2. check  — slow: scan files with torch.load and save bad list
  3. remove — delete files listed in the text file from 'check'

Usage:
  # Fast: remove near-zero files immediately
  python scripts/demo/remove_small_pt.py small data/media/robofac/pointer_memory
  python scripts/demo/remove_small_pt.py small data/media/robofac/pointer_memory --dry-run

  # Slow: validate with torch.load, save list for review
  python scripts/demo/remove_small_pt.py check data/media/robofac/pointer_memory

  # Delete the files found by 'check'
  python scripts/demo/remove_small_pt.py remove bad_pt_files.txt
"""

import argparse
from pathlib import Path

REQUIRED_KEYS = {"pointer_timestamps", "pointer_memory_embeds", "pointer_positions"}
DEFAULT_OUTPUT = "bad_pt_files.txt"


def _check_pt(path_str):
    """Return (path, reason) if the file is bad, else (path, None)."""
    import torch

    p = Path(path_str)

    try:
        data = torch.load(p, weights_only=True)
    except Exception as e:
        return (path_str, f"unreadable ({type(e).__name__})")

    if not isinstance(data, dict):
        return (path_str, "not a dict")
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return (path_str, f"missing keys: {missing}")

    return (path_str, None)


def cmd_small(args):
    """Fast pass: remove files below size threshold (no torch.load)."""
    pt_files = sorted(Path(args.folder).rglob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {args.folder}")

    small_files = [(p, p.stat().st_size) for p in pt_files if p.stat().st_size < args.threshold]

    if not small_files:
        print("No files below threshold.")
        return

    print(f"\n{len(small_files)} file(s) below {args.threshold} bytes:")
    for p, size in small_files:
        print(f"  {size:>6d} B  {p}")

    if args.dry_run:
        print("\n[dry-run] No files deleted.")
        return

    for p, _ in small_files:
        p.unlink()
    print(f"\nDeleted {len(small_files)} file(s).")


def cmd_check(args):
    """Slow pass: validate files with torch.load, save bad list."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    pt_files = sorted(Path(args.folder).rglob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {args.folder}")

    if not pt_files:
        return

    bad_files = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_check_pt, str(p)): p for p in pt_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Checking"):
            path_str, reason = fut.result()
            if reason is not None:
                bad_files.append((path_str, reason))

    if not bad_files:
        print("All files are valid.")
        return

    bad_files.sort()
    print(f"\n{len(bad_files)} bad file(s):")
    for path_str, reason in bad_files:
        print(f"  {reason:>40s}  {path_str}")

    output = Path(args.output)
    with open(output, "w") as f:
        for path_str, reason in bad_files:
            f.write(f"{path_str}\t{reason}\n")
    print(f"\nSaved list to {output}")
    print(f"Review it, then run:  python {__file__} remove {output}")


def cmd_remove(args):
    """Delete files listed in a text file from the 'check' step."""
    list_path = Path(args.list_file)
    if not list_path.exists():
        print(f"File not found: {list_path}")
        return

    lines = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
    paths = [l.split("\t")[0] for l in lines]
    print(f"{len(paths)} file(s) to delete (from {list_path})")

    deleted = 0
    for p_str in paths:
        p = Path(p_str)
        if p.exists():
            p.unlink()
            deleted += 1
        else:
            print(f"  skipped (not found): {p_str}")

    print(f"Deleted {deleted} file(s).")


def main():
    parser = argparse.ArgumentParser(description="Find and remove broken .pt files")
    sub = parser.add_subparsers(dest="command", required=True)

    p_small = sub.add_parser("small", help="Fast: remove files below size threshold")
    p_small.add_argument("folder", help="Path to pointer_memory folder")
    p_small.add_argument("--threshold", type=int, default=1024, help="Size threshold in bytes (default: 1024)")
    p_small.add_argument("--dry-run", action="store_true", help="List files without deleting")

    p_check = sub.add_parser("check", help="Slow: validate with torch.load, save bad list")
    p_check.add_argument("folder", help="Path to pointer_memory folder")
    p_check.add_argument("--workers", type=int, default=8)
    p_check.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")

    p_remove = sub.add_parser("remove", help="Delete files listed in a text file")
    p_remove.add_argument("list_file", help="Text file from the check step")

    args = parser.parse_args()
    {"small": cmd_small, "check": cmd_check, "remove": cmd_remove}[args.command](args)


if __name__ == "__main__":
    main()
