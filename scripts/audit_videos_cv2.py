#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path

import cv2


ROOT = Path.home() / "PhysMelt"
VIDEO_DIR = ROOT / "data/raw/NIST_RHF/MPM_AVIs"

OUT_CSV = ROOT / "data/manifests/video_manifest_cv2.csv"
OUT_JSON = ROOT / "data/manifests/video_summary_cv2.json"


def video_number(path):
    match = re.search(r"P(\d+)", path.stem)
    return int(match.group(1)) if match else 999999


def audit_video(path):
    cap = cv2.VideoCapture(str(path))

    row = {
        "video_id": path.stem,
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024**2, 3),
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "reported_frames": 0,
        "decoded_frames": 0,
        "duration_s": 0.0,
        "frame_shape": "",
        "dtype": "",
        "pixel_min": "",
        "pixel_max": "",
        "channels_equal": "",
        "status": "error",
        "error": "",
    }

    if not cap.isOpened():
        row["error"] = "OpenCV could not open the video"
        return row

    row["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    row["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    row["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    row["reported_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    decoded_frames = 0
    first_frame = None

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if first_frame is None:
            first_frame = frame.copy()

        decoded_frames += 1

    cap.release()

    row["decoded_frames"] = decoded_frames

    if row["fps"] > 0:
        row["duration_s"] = round(decoded_frames / row["fps"], 6)

    if first_frame is not None:
        row["frame_shape"] = "x".join(str(value) for value in first_frame.shape)
        row["dtype"] = str(first_frame.dtype)
        row["pixel_min"] = int(first_frame.min())
        row["pixel_max"] = int(first_frame.max())

        if first_frame.ndim == 3 and first_frame.shape[2] == 3:
            b, g, r = cv2.split(first_frame)
            row["channels_equal"] = bool(
                (b == g).all() and (g == r).all()
            )
        else:
            row["channels_equal"] = True

    if decoded_frames > 0:
        row["status"] = "ok"
    else:
        row["error"] = "No frames could be decoded"

    return row


def main():
    videos = sorted(VIDEO_DIR.glob("*.avi"), key=video_number)

    print("Video directory:", VIDEO_DIR)
    print("Videos found:", len(videos))
    print()

    rows = []

    for index, path in enumerate(videos, start=1):
        print(f"[{index:02d}/{len(videos):02d}] {path.name}", flush=True)

        try:
            row = audit_video(path)
        except Exception as exc:
            row = {
                "video_id": path.stem,
                "filename": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / 1024**2, 3),
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "reported_frames": 0,
                "decoded_frames": 0,
                "duration_s": 0.0,
                "frame_shape": "",
                "dtype": "",
                "pixel_min": "",
                "pixel_max": "",
                "channels_equal": "",
                "status": "error",
                "error": str(exc),
            }

        rows.append(row)

        print(
            f"    status={row['status']}, "
            f"size={row['width']}x{row['height']}, "
            f"fps={row['fps']}, "
            f"reported={row['reported_frames']}, "
            f"decoded={row['decoded_frames']}"
        )

        if row["error"]:
            print("    ERROR:", row["error"])

    fieldnames = list(rows[0].keys()) if rows else []

    if rows:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] == "ok"]

    summary = {
        "video_directory": str(VIDEO_DIR),
        "video_count": len(rows),
        "valid_video_count": len(valid_rows),
        "failed_video_count": len(rows) - len(valid_rows),
        "total_decoded_frames": sum(
            row["decoded_frames"] for row in valid_rows
        ),
        "total_reported_frames": sum(
            row["reported_frames"] for row in valid_rows
        ),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "total_size_gb": round(
            sum(row["size_bytes"] for row in rows) / 1024**3,
            3,
        ),
        "resolutions": sorted({
            f"{row['width']}x{row['height']}"
            for row in valid_rows
        }),
        "frame_rates": sorted({
            row["fps"]
            for row in valid_rows
        }),
        "frame_dtypes": sorted({
            row["dtype"]
            for row in valid_rows
        }),
        "all_channels_equal": all(
            row["channels_equal"] is True
            for row in valid_rows
        ) if valid_rows else False,
    }

    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print()
    print("CSV:", OUT_CSV)
    print("JSON:", OUT_JSON)
    print("Аудит завершён. Терминал остаётся открытым.")


if __name__ == "__main__":
    main()
