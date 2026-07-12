#!/usr/bin/env python3

import csv
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path.home() / "PhysMelt"
VIDEO_DIR = ROOT / "data/raw/NIST_RHF/MPM_AVIs"
OUT_CSV = ROOT / "data/manifests/video_manifest.csv"
OUT_JSON = ROOT / "data/manifests/video_summary.json"


def parse_rate(value):
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path):
    command = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        (
            "stream=codec_name,codec_long_name,width,height,pix_fmt,"
            "bits_per_raw_sample,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration"
        ),
        "-of", "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])

    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]

    fps = parse_rate(
        stream.get("avg_frame_rate")
        or stream.get("r_frame_rate")
    )

    frame_count = (
        stream.get("nb_read_frames")
        or stream.get("nb_frames")
    )

    try:
        frame_count = int(frame_count)
    except (TypeError, ValueError):
        frame_count = None

    try:
        duration = float(stream.get("duration"))
    except (TypeError, ValueError):
        duration = None

    if duration is None and frame_count is not None and fps:
        duration = frame_count / fps

    return {
        "video_id": path.stem,
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024**2, 3),
        "codec_name": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pixel_format": stream.get("pix_fmt"),
        "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
        "fps": round(fps, 6) if fps is not None else None,
        "frame_count": frame_count,
        "duration_s": round(duration, 6) if duration is not None else None,
        "status": "ok",
        "error": "",
    }


def main():
    if not VIDEO_DIR.exists():
        print(f"ERROR: video directory not found: {VIDEO_DIR}", file=sys.stderr)
        return 1

    videos = sorted(VIDEO_DIR.glob("*.avi"))

    if not videos:
        print(f"ERROR: no AVI files found in {VIDEO_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} AVI files")
    print(f"Input:  {VIDEO_DIR}")
    print(f"Output: {OUT_CSV}")
    print()

    rows = []

    for index, path in enumerate(videos, start=1):
        print(f"[{index:02d}/{len(videos):02d}] {path.name}", flush=True)

        try:
            row = probe_video(path)
            print(
                f"    {row['width']}x{row['height']}, "
                f"{row['fps']} fps, "
                f"{row['frame_count']} frames, "
                f"{row['pixel_format']}"
            )
        except Exception as exc:
            row = {
                "video_id": path.stem,
                "filename": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / 1024**2, 3),
                "codec_name": None,
                "codec_long_name": None,
                "width": None,
                "height": None,
                "pixel_format": None,
                "bits_per_raw_sample": None,
                "fps": None,
                "frame_count": None,
                "duration_s": None,
                "status": "error",
                "error": str(exc),
            }
            print(f"    ERROR: {exc}", file=sys.stderr)

        rows.append(row)

    fieldnames = [
        "video_id",
        "filename",
        "path",
        "size_bytes",
        "size_mb",
        "codec_name",
        "codec_long_name",
        "width",
        "height",
        "pixel_format",
        "bits_per_raw_sample",
        "fps",
        "frame_count",
        "duration_s",
        "status",
        "error",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["status"] == "ok"]
    total_frames = sum(
        row["frame_count"]
        for row in valid
        if row["frame_count"] is not None
    )
    total_size = sum(row["size_bytes"] for row in rows)

    resolutions = sorted({
        f"{row['width']}x{row['height']}"
        for row in valid
    })
    frame_rates = sorted({
        row["fps"]
        for row in valid
        if row["fps"] is not None
    })
    pixel_formats = sorted({
        row["pixel_format"]
        for row in valid
        if row["pixel_format"]
    })

    summary = {
        "video_directory": str(VIDEO_DIR),
        "video_count": len(rows),
        "valid_video_count": len(valid),
        "failed_video_count": len(rows) - len(valid),
        "total_frames": total_frames,
        "total_size_bytes": total_size,
        "total_size_gb": round(total_size / 1024**3, 3),
        "resolutions": resolutions,
        "frame_rates": frame_rates,
        "pixel_formats": pixel_formats,
    }

    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print()
    print(f"CSV:  {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")

    return 0 if summary["failed_video_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
