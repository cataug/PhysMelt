#!/usr/bin/env python3

import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path.home() / "PhysMelt"
FEATURE_DIR = ROOT / "data/features/frame_features"
MANIFEST_DIR = ROOT / "data/manifests"

OUTPUT_CSV = FEATURE_DIR / "all_frame_features.csv"
SUMMARY_JSON = FEATURE_DIR / "dataset_summary.json"
PARTITIONS_CSV = MANIFEST_DIR / "video_partitions.csv"


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    missing = []

    for number in range(1, 56):
        video_id = f"RHF_MPM_P{number:02d}"
        summary_path = FEATURE_DIR / f"{video_id}_summary.json"
        features_path = FEATURE_DIR / f"{video_id}_features.csv"

        if not summary_path.exists() or not features_path.exists():
            missing.append(video_id)
            continue

        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        summary["features_path"] = str(features_path)
        summaries.append(summary)

    print("Valid feature sets:", len(summaries))
    print("Missing feature sets:", len(missing))

    if missing:
        print("Missing:", ", ".join(missing))
        print("Dataset was not merged because some videos are missing.")
        return

    summaries.sort(key=lambda item: item["video_id"])

    video_ids = [item["video_id"] for item in summaries]

    shuffled_ids = video_ids.copy()
    random.Random(42).shuffle(shuffled_ids)

    split_lookup = {}
    fold_lookup = {}

    for index, video_id in enumerate(shuffled_ids):
        if index < 33:
            split_name = "train"
        elif index < 44:
            split_name = "validation"
        else:
            split_name = "test"

        split_lookup[video_id] = split_name
        fold_lookup[video_id] = index % 5

    with PARTITIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["video_id", "split", "fold"],
        )
        writer.writeheader()

        for video_id in sorted(video_ids):
            writer.writerow({
                "video_id": video_id,
                "split": split_lookup[video_id],
                "fold": fold_lookup[video_id],
            })

    expected_header = None
    total_rows = 0
    active_rows = 0
    inactive_rows = 0
    per_video_rows = {}

    temporary_output = OUTPUT_CSV.with_suffix(".csv.tmp")

    with temporary_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_handle:

        writer = None

        for position, summary in enumerate(summaries, start=1):
            video_id = summary["video_id"]
            features_path = Path(summary["features_path"])

            print(
                f"[{position:02d}/55] Merging {features_path.name}",
                flush=True,
            )

            video_rows = 0

            with features_path.open(
                newline="",
                encoding="utf-8",
            ) as input_handle:

                reader = csv.DictReader(input_handle)

                if reader.fieldnames is None:
                    print("Missing header:", features_path)
                    return

                if expected_header is None:
                    expected_header = list(reader.fieldnames)

                    output_fields = (
                        expected_header
                        + ["split", "fold"]
                    )

                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=output_fields,
                    )
                    writer.writeheader()

                elif list(reader.fieldnames) != expected_header:
                    print("Header mismatch:", features_path)
                    return

                for row in reader:
                    row["split"] = split_lookup[video_id]
                    row["fold"] = fold_lookup[video_id]

                    writer.writerow(row)

                    video_rows += 1
                    total_rows += 1

                    if row.get("active_candidate") == "1":
                        active_rows += 1
                    else:
                        inactive_rows += 1

            per_video_rows[video_id] = video_rows

    temporary_output.replace(OUTPUT_CSV)

    active_fractions = [
        item["active_candidate_fraction"]
        for item in summaries
    ]

    frame_counts = [
        item["frame_count"]
        for item in summaries
    ]

    split_counts = {
        "train": sum(
            split_lookup[video_id] == "train"
            for video_id in video_ids
        ),
        "validation": sum(
            split_lookup[video_id] == "validation"
            for video_id in video_ids
        ),
        "test": sum(
            split_lookup[video_id] == "test"
            for video_id in video_ids
        ),
    }

    fold_counts = {
        str(fold): sum(
            fold_lookup[video_id] == fold
            for video_id in video_ids
        )
        for fold in range(5)
    }

    dataset_summary = {
        "video_count": len(summaries),
        "total_frames": total_rows,
        "expected_total_frames": 82390,
        "all_videos_have_1498_frames": all(
            count == 1498 for count in frame_counts
        ),
        "active_frames": active_rows,
        "inactive_frames": inactive_rows,
        "active_fraction": (
            active_rows / total_rows
            if total_rows
            else 0.0
        ),
        "per_video_active_fraction_min": min(active_fractions),
        "per_video_active_fraction_median": statistics.median(
            active_fractions
        ),
        "per_video_active_fraction_max": max(active_fractions),
        "split_video_counts": split_counts,
        "fold_video_counts": fold_counts,
        "feature_table": str(OUTPUT_CSV),
        "partitions_file": str(PARTITIONS_CSV),
        "per_video_frame_counts": per_video_rows,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(dataset_summary, output, indent=2)

    print()
    print("=== DATASET SUMMARY ===")
    print(json.dumps(dataset_summary, indent=2))
    print()
    print("Combined features:", OUTPUT_CSV)
    print("Partitions:", PARTITIONS_CSV)
    print("Summary:", SUMMARY_JSON)
    print("Dataset construction finished. The terminal remains open.")


if __name__ == "__main__":
    main()
