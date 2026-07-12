#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path.home() / "PhysMelt"
EXTRACTOR = ROOT / "scripts/extract_frame_features.py"
OUTPUT_DIR = ROOT / "data/features/frame_features"
LOG_DIR = ROOT / "logs/extraction"


def completed_correctly(video_id):
    summary_path = OUTPUT_DIR / f"RHF_MPM_{video_id}_summary.json"
    features_path = OUTPUT_DIR / f"RHF_MPM_{video_id}_features.csv"

    if not summary_path.exists() or not features_path.exists():
        return False

    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        return (
            summary.get("frame_count") == 1498
            and features_path.stat().st_size > 0
        )
    except Exception:
        return False


def run_video(video_id):
    log_path = LOG_DIR / f"extract_{video_id}.txt"

    command = [
        sys.executable,
        str(EXTRACTOR),
        "--video",
        video_id,
    ]

    print()
    print("=" * 72)
    print(f"START {video_id}")
    print("Command:", " ".join(command))
    print("Log:", log_path)
    print("=" * 72)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()

        return_code = process.wait()

    if return_code == 0 and completed_correctly(video_id):
        print(f"DONE {video_id}")
        return True

    print(f"FAILED {video_id}, return code {return_code}")
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not EXTRACTOR.exists():
        print("Extractor was not found:", EXTRACTOR)
        return

    completed = []
    skipped = []
    failed = []

    for number in range(1, 56):
        video_id = f"P{number:02d}"

        if completed_correctly(video_id):
            print(f"[{number:02d}/55] SKIP {video_id}: already complete")
            skipped.append(video_id)
            continue

        print(f"[{number:02d}/55] PROCESS {video_id}")

        if run_video(video_id):
            completed.append(video_id)
        else:
            failed.append(video_id)

    print()
    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    print("Newly completed:", len(completed))
    print("Already complete:", len(skipped))
    print("Failed:", len(failed))

    if failed:
        print("Failed videos:", ", ".join(failed))
    else:
        print("All 55 videos have valid feature files.")

    print()
    print("Batch processing finished. The terminal remains open.")


if __name__ == "__main__":
    main()
