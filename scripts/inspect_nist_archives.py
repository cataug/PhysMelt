#!/usr/bin/env python3

import csv
import io
import json
import re
import statistics
import zipfile
from pathlib import Path


ROOT = Path.home() / "PhysMelt/data/raw/NIST_RHF"
REPORT = (
    Path.home()
    / "PhysMelt/data/manifests/nist_archive_inventory.json"
)


def is_number(value):
    try:
        float(value.strip())
        return True
    except Exception:
        return False


def classify_csv(name):
    lower = name.lower()

    if "daq_rhf_" in lower:
        return "analysis_results"

    if re.search(r"(^|/)daq_p\d+", lower):
        return "encoder_feedback"

    if re.search(r"(^|/)rhf_p\d+", lower):
        return "command"

    return "other_csv"


def inspect_csv(zip_handle, member_name):
    total_nonempty_rows = 0
    first_rows = []

    with zip_handle.open(member_name) as raw_handle:
        text_handle = io.TextIOWrapper(
            raw_handle,
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        )

        reader = csv.reader(text_handle)

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue

            total_nonempty_rows += 1

            if len(first_rows) < 2:
                first_rows.append(row)

    first_row = first_rows[0] if first_rows else []
    second_row = first_rows[1] if len(first_rows) > 1 else []

    has_header = (
        bool(first_row)
        and any(
            cell.strip() and not is_number(cell)
            for cell in first_row
        )
    )

    data_rows = (
        total_nonempty_rows - 1
        if has_header
        else total_nonempty_rows
    )

    return {
        "member": member_name,
        "total_nonempty_rows": total_nonempty_rows,
        "data_rows": data_rows,
        "has_header": has_header,
        "first_row": first_row,
        "second_row": second_row,
    }


def summarize_group(items):
    row_counts = [
        item["data_rows"]
        for item in items
    ]

    return {
        "file_count": len(items),
        "row_count_min": min(row_counts) if row_counts else None,
        "row_count_median": (
            statistics.median(row_counts)
            if row_counts
            else None
        ),
        "row_count_max": max(row_counts) if row_counts else None,
        "representative_file": (
            items[0]["member"] if items else None
        ),
        "representative_first_row": (
            items[0]["first_row"] if items else None
        ),
        "representative_second_row": (
            items[0]["second_row"] if items else None
        ),
    }


def main():
    print("NIST RHF directory:")
    print(ROOT)
    print()

    if not ROOT.exists():
        print("Directory does not exist.")
        return

    archives = sorted(ROOT.rglob("*.zip"))

    print("ZIP archives found:", len(archives))

    if not archives:
        print("No ZIP archives were found.")
        print()
        print("Existing top-level files:")

        for path in sorted(ROOT.iterdir()):
            print(" ", path.name)

        return

    report = {
        "root": str(ROOT),
        "archives": [],
    }

    for archive_index, archive_path in enumerate(
        archives,
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"[{archive_index}/{len(archives)}] "
            f"{archive_path.name}"
        )
        print(
            "Size:",
            round(archive_path.stat().st_size / 1024**2, 2),
            "MB",
        )

        archive_report = {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "valid_zip": False,
            "member_count": 0,
            "groups": {},
            "error": "",
        }

        try:
            with zipfile.ZipFile(archive_path, "r") as zip_handle:
                bad_member = zip_handle.testzip()

                if bad_member is not None:
                    archive_report["error"] = (
                        f"CRC failure in {bad_member}"
                    )
                    print("CRC ERROR:", bad_member)
                    report["archives"].append(archive_report)
                    continue

                archive_report["valid_zip"] = True

                members = [
                    name
                    for name in zip_handle.namelist()
                    if not name.endswith("/")
                ]

                archive_report["member_count"] = len(members)

                print("Members:", len(members))

                csv_members = [
                    name
                    for name in members
                    if name.lower().endswith(".csv")
                ]

                print("CSV files:", len(csv_members))

                grouped = {
                    "analysis_results": [],
                    "encoder_feedback": [],
                    "command": [],
                    "other_csv": [],
                }

                for csv_index, member_name in enumerate(
                    csv_members,
                    start=1,
                ):
                    group_name = classify_csv(member_name)

                    print(
                        f"  [{csv_index:03d}/{len(csv_members):03d}] "
                        f"{group_name}: {member_name}",
                        flush=True,
                    )

                    try:
                        information = inspect_csv(
                            zip_handle,
                            member_name,
                        )
                        grouped[group_name].append(information)
                    except Exception as exc:
                        grouped[group_name].append({
                            "member": member_name,
                            "data_rows": 0,
                            "has_header": False,
                            "first_row": [],
                            "second_row": [],
                            "error": str(exc),
                        })
                        print("      ERROR:", exc)

                for group_name, items in grouped.items():
                    if not items:
                        continue

                    summary = summarize_group(items)
                    archive_report["groups"][group_name] = {
                        "summary": summary,
                        "files": items,
                    }

                    print()
                    print("GROUP:", group_name)
                    print("  Files:", summary["file_count"])
                    print(
                        "  Data rows min/median/max:",
                        summary["row_count_min"],
                        summary["row_count_median"],
                        summary["row_count_max"],
                    )
                    print(
                        "  Example:",
                        summary["representative_file"],
                    )
                    print(
                        "  First row:",
                        summary["representative_first_row"],
                    )
                    print(
                        "  Second row:",
                        summary["representative_second_row"],
                    )

        except zipfile.BadZipFile as exc:
            archive_report["error"] = str(exc)
            print("BAD ZIP:", exc)

        except Exception as exc:
            archive_report["error"] = str(exc)
            print("ERROR:", exc)

        report["archives"].append(archive_report)

    extracted_csvs = sorted(ROOT.rglob("*.csv"))

    report["already_extracted_csv_count"] = len(extracted_csvs)
    report["already_extracted_csvs"] = [
        str(path)
        for path in extracted_csvs
    ]

    print()
    print("=" * 78)
    print("Already extracted CSV files:", len(extracted_csvs))

    for path in extracted_csvs[:20]:
        print(" ", path)

    if len(extracted_csvs) > 20:
        print("  ...")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    temporary_report = REPORT.with_suffix(".json.tmp")

    with temporary_report.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)

    temporary_report.replace(REPORT)

    print()
    print("Inventory saved:")
    print(REPORT)
    print()
    print("Inspection finished. The terminal remains open.")


if __name__ == "__main__":
    main()
