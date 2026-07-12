#!/usr/bin/env python3

import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path.home() / "PhysMelt/data/raw/NIST_RHF"
OUT = Path.home() / "PhysMelt/data/manifests/nist_sync_report.json"

EXPECTED_FRAMES = 1498


def find_archive(name):
    matches = [
        path for path in ROOT.rglob("*.zip")
        if path.name.lower() == name.lower()
    ]
    return matches[0] if matches else None


def numeric_rows(zip_handle, member):
    rows = []

    with zip_handle.open(member) as raw:
        text = io.TextIOWrapper(
            raw,
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        )

        for row in csv.reader(text):
            if not row:
                continue

            try:
                values = [float(cell.strip()) for cell in row]
            except ValueError:
                continue

            rows.append(values)

    return rows


def rising_count(values, target=None):
    count = 0
    previous = False

    for value in values:
        current = (
            value != 0
            if target is None
            else abs(value - target) < 1e-9
        )

        if current and not previous:
            count += 1

        previous = current

    return count


def inspect_encoder(archive_path):
    results = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [
            name for name in zf.namelist()
            if re.search(
                r"DAQ_RHF_P\d+_layer0001\.csv$",
                name,
                flags=re.IGNORECASE,
            )
            and "T80" not in name.upper()
        ]

        members.sort()

        print("Encoder CSV files:", len(members))
        print()

        for index, member in enumerate(members, start=1):
            rows = numeric_rows(zf, member)

            if not rows:
                print("EMPTY:", member)
                continue

            column_count = min(len(row) for row in rows)
            rows = [row[:column_count] for row in rows]

            candidates = []

            for column in range(column_count):
                values = [row[column] for row in rows]

                equal_two = sum(
                    abs(value - 2.0) < 1e-9
                    for value in values
                )
                nonzero = sum(
                    abs(value) > 1e-12
                    for value in values
                )
                rising_two = rising_count(values, target=2.0)
                rising_nonzero = rising_count(values)

                variants = {
                    "equal_2": equal_two,
                    "nonzero": nonzero,
                    "rising_2": rising_two,
                    "rising_nonzero": rising_nonzero,
                }

                for mode, count in variants.items():
                    candidates.append({
                        "column": column,
                        "mode": mode,
                        "count": count,
                        "distance": abs(count - EXPECTED_FRAMES),
                    })

            candidates.sort(
                key=lambda item: (
                    item["distance"],
                    item["column"],
                    item["mode"],
                )
            )

            best = candidates[0]

            result = {
                "member": member,
                "rows": len(rows),
                "columns": column_count,
                "best_trigger_column": best["column"],
                "best_trigger_mode": best["mode"],
                "best_trigger_count": best["count"],
                "distance_from_1498": best["distance"],
                "first_row": rows[0],
                "second_row": rows[1] if len(rows) > 1 else [],
            }

            results.append(result)

            print(
                f"[{index:02d}/{len(members):02d}] "
                f"{Path(member).name}: "
                f"rows={len(rows)}, cols={column_count}, "
                f"trigger_col={best['column']}, "
                f"mode={best['mode']}, "
                f"events={best['count']}"
            )

            if index == 1:
                print()
                print("P01 column diagnostics:")

                for column in range(column_count):
                    values = [row[column] for row in rows]
                    unique = Counter(values)

                    print(
                        f"  column {column}: "
                        f"min={min(values):.6g}, "
                        f"max={max(values):.6g}, "
                        f"nonzero={sum(abs(v) > 1e-12 for v in values)}, "
                        f"equal_2={sum(abs(v - 2) < 1e-9 for v in values)}, "
                        f"rising_nonzero={rising_count(values)}, "
                        f"common={unique.most_common(5)}"
                    )

                print()

    exact = sum(
        item["best_trigger_count"] == EXPECTED_FRAMES
        for item in results
    )

    print()
    print("Encoder parts with exactly 1498 detected events:")
    print(f"{exact}/{len(results)}")

    return results


def inspect_analysis(archive_path):
    if archive_path is None:
        print()
        print("RHF_Analysis_Results.zip was not found.")
        return []

    results = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [
            name for name in zf.namelist()
            if name.lower().endswith(".csv")
        ]
        members.sort()

        print()
        print("Analysis-results CSV files:", len(members))

        for index, member in enumerate(members, start=1):
            rows = numeric_rows(zf, member)

            column_count = (
                min(len(row) for row in rows)
                if rows else 0
            )

            result = {
                "member": member,
                "rows": len(rows),
                "columns": column_count,
                "first_row": rows[0] if rows else [],
                "second_row": rows[1] if len(rows) > 1 else [],
            }
            results.append(result)

            if (
                index <= 3
                or index > len(members) - 3
            ):
                print(
                    f"[{index:02d}/{len(members):02d}] "
                    f"{Path(member).name}: "
                    f"rows={len(rows)}, cols={column_count}"
                )

        if results:
            counts = sorted(item["rows"] for item in results)
            columns = sorted(set(item["columns"] for item in results))

            print()
            print(
                "Analysis rows min/median/max:",
                counts[0],
                counts[len(counts) // 2],
                counts[-1],
            )
            print("Column counts:", columns)
            print("Example member:", results[0]["member"])
            print("Example first row:", results[0]["first_row"])
            print("Example second row:", results[0]["second_row"])

    return results


def main():
    encoder_archive = find_archive("RHF_Encoder.zip")
    analysis_archive = find_archive("RHF_Analysis_Results.zip")

    print("Encoder archive:", encoder_archive)
    print("Analysis archive:", analysis_archive)
    print()

    if encoder_archive is None:
        print("RHF_Encoder.zip was not found.")
        return

    encoder_results = inspect_encoder(encoder_archive)
    analysis_results = inspect_analysis(analysis_archive)

    report = {
        "encoder_archive": str(encoder_archive),
        "analysis_archive": (
            str(analysis_archive)
            if analysis_archive else None
        ),
        "expected_video_frames_per_part": EXPECTED_FRAMES,
        "encoder": encoder_results,
        "analysis_results": analysis_results,
    }

    temporary = OUT.with_suffix(".json.tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    temporary.replace(OUT)

    print()
    print("Report saved:")
    print(OUT)
    print()
    print("Inspection finished. The terminal remains open.")


if __name__ == "__main__":
    main()
