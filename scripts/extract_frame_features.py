#!/usr/bin/env python3

import argparse
import csv
import json
import math
import struct
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path.home() / "PhysMelt"
VIDEO_DIR = ROOT / "data/raw/NIST_RHF/MPM_AVIs"
OUTPUT_DIR = ROOT / "data/features/frame_features"

PHYSICAL_FPS = 20000.0
WIDTH = 120
HEIGHT = 120
CHANNELS = 3
ROW_STRIDE = ((WIDTH * CHANNELS + 3) // 4) * 4
FRAME_BYTES = ROW_STRIDE * HEIGHT
ROLLING_WINDOW = 11

Y_GRID, X_GRID = np.indices((HEIGHT, WIDTH), dtype=np.float64)


def iter_chunks(handle, start, end):
    position = start

    while position + 8 <= end:
        handle.seek(position)
        chunk_id = handle.read(4)
        size_raw = handle.read(4)

        if len(chunk_id) != 4 or len(size_raw) != 4:
            break

        size = struct.unpack("<I", size_raw)[0]
        data_start = position + 8
        data_end = min(data_start + size, end)

        yield chunk_id, size, data_start, data_end

        next_position = data_start + size + (size % 2)

        if next_position <= position:
            break

        position = next_position


def find_movi(handle, file_size):
    for chunk_id, size, data_start, data_end in iter_chunks(
        handle, 12, file_size
    ):
        if chunk_id != b"LIST":
            continue

        handle.seek(data_start)
        list_type = handle.read(4)

        if list_type == b"movi":
            return data_start + 4, data_end

    raise RuntimeError("LIST movi section was not found")


def collect_frame_chunks(handle, start, end, output):
    for chunk_id, size, data_start, data_end in iter_chunks(
        handle, start, end
    ):
        if chunk_id == b"LIST":
            handle.seek(data_start)
            list_type = handle.read(4)

            if list_type in {b"rec ", b"movi"}:
                collect_frame_chunks(
                    handle,
                    data_start + 4,
                    data_end,
                    output,
                )

        elif len(chunk_id) == 4 and chunk_id[2:4] in {b"db", b"dc"}:
            output.append((data_start, size))


def decode_gray_frame(handle, offset, size):
    handle.seek(offset)
    payload = handle.read(size)

    if len(payload) < FRAME_BYTES:
        raise RuntimeError(
            f"Short frame payload: {len(payload)} < {FRAME_BYTES}"
        )

    raw = np.frombuffer(
        payload[:FRAME_BYTES],
        dtype=np.uint8,
    ).reshape(HEIGHT, ROW_STRIDE)

    bgr = raw[:, :WIDTH * CHANNELS].reshape(
        HEIGHT,
        WIDTH,
        CHANNELS,
    )

    # Positive DIB height means bottom-up storage.
    bgr = bgr[::-1]

    # All three stored channels are identical in this dataset.
    return bgr[:, :, 0].copy()


def histogram_percentile(histogram, quantile):
    cumulative = np.cumsum(histogram)
    target = quantile * cumulative[-1]
    return int(np.searchsorted(cumulative, target, side="left"))


def otsu_threshold(histogram):
    histogram = histogram.astype(np.float64)
    total = histogram.sum()

    if total <= 0:
        return 0

    levels = np.arange(256, dtype=np.float64)
    cumulative_weight = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * levels)
    total_sum = cumulative_sum[-1]

    denominator = cumulative_weight * (total - cumulative_weight)
    valid = denominator > 0

    between = np.zeros(256, dtype=np.float64)
    numerator = (
        total_sum * cumulative_weight - cumulative_sum * total
    )

    between[valid] = (
        numerator[valid] ** 2
        / denominator[valid]
    )

    return int(np.argmax(between))


def weighted_geometry(image):
    weights = image.astype(np.float64)
    weight_sum = weights.sum()

    empty = {
        "centroid_x": math.nan,
        "centroid_y": math.nan,
        "major_sigma": 0.0,
        "minor_sigma": 0.0,
        "eccentricity": 0.0,
        "orientation_deg": 0.0,
        "tail_skewness": 0.0,
        "abs_tail_skewness": 0.0,
        "outer_intensity_fraction": 0.0,
    }

    if weight_sum <= 0:
        return empty

    centroid_x = float((weights * X_GRID).sum() / weight_sum)
    centroid_y = float((weights * Y_GRID).sum() / weight_sum)

    dx = X_GRID - centroid_x
    dy = Y_GRID - centroid_y

    var_x = float((weights * dx * dx).sum() / weight_sum)
    var_y = float((weights * dy * dy).sum() / weight_sum)
    cov_xy = float((weights * dx * dy).sum() / weight_sum)

    covariance = np.array(
        [[var_x, cov_xy], [cov_xy, var_y]],
        dtype=np.float64,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    major_variance = max(float(eigenvalues[order[0]]), 0.0)
    minor_variance = max(float(eigenvalues[order[1]]), 0.0)

    major_sigma = math.sqrt(major_variance)
    minor_sigma = math.sqrt(minor_variance)

    major_vector = eigenvectors[:, order[0]]
    vx = float(major_vector[0])
    vy = float(major_vector[1])

    orientation_deg = math.degrees(math.atan2(vy, vx))

    if major_variance > 0:
        eccentricity = math.sqrt(
            max(0.0, 1.0 - minor_variance / major_variance)
        )
    else:
        eccentricity = 0.0

    projection_major = dx * vx + dy * vy
    projection_minor = -dx * vy + dy * vx

    if major_sigma > 1e-9:
        tail_skewness = float(
            (
                weights * projection_major ** 3
            ).sum()
            / (weight_sum * major_sigma ** 3)
        )
    else:
        tail_skewness = 0.0

    major_scale = max(3.0 * major_sigma, 1.0)
    minor_scale = max(3.0 * minor_sigma, 1.0)

    elliptical_distance = (
        (projection_major / major_scale) ** 2
        + (projection_minor / minor_scale) ** 2
    )

    outer_mask = elliptical_distance > 1.0
    outer_intensity_fraction = float(
        weights[outer_mask].sum() / weight_sum
    )

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "major_sigma": major_sigma,
        "minor_sigma": minor_sigma,
        "eccentricity": eccentricity,
        "orientation_deg": orientation_deg,
        "tail_skewness": tail_skewness,
        "abs_tail_skewness": abs(tail_skewness),
        "outer_intensity_fraction": outer_intensity_fraction,
    }


def extract_features(image):
    flat = image.ravel()
    histogram = np.bincount(flat, minlength=256)

    minimum = int(flat.min())
    maximum = int(flat.max())
    mean = float(flat.mean())
    standard_deviation = float(flat.std())
    total_intensity = int(flat.sum())

    p95 = histogram_percentile(histogram, 0.95)
    p99 = histogram_percentile(histogram, 0.99)
    p999 = histogram_percentile(histogram, 0.999)

    probabilities = histogram[histogram > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    otsu = otsu_threshold(histogram)
    segmentation_threshold = max(8, otsu)

    mask = image >= segmentation_threshold
    area = int(mask.sum())

    if area > 0:
        ys, xs = np.nonzero(mask)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        bbox_fill = float(area / (bbox_width * bbox_height))
        threshold_centroid_x = float(xs.mean())
        threshold_centroid_y = float(ys.mean())
    else:
        bbox_width = 0
        bbox_height = 0
        bbox_fill = 0.0
        threshold_centroid_x = math.nan
        threshold_centroid_y = math.nan

    geometry = weighted_geometry(image)

    saturation_fraction = float(np.mean(image == 255))
    nonzero_fraction = float(np.mean(image > 0))
    bright_fraction_64 = float(np.mean(image >= 64))
    bright_fraction_128 = float(np.mean(image >= 128))
    bright_fraction_200 = float(np.mean(image >= 200))

    effective_area = (
        float(total_intensity / maximum)
        if maximum > 0
        else 0.0
    )

    active_candidate = int(
        maximum >= 20
        and area >= 5
        and total_intensity >= 100
    )

    result = {
        "pixel_min": minimum,
        "pixel_max": maximum,
        "mean_intensity": mean,
        "std_intensity": standard_deviation,
        "total_intensity": total_intensity,
        "effective_area": effective_area,
        "p95": p95,
        "p99": p99,
        "p999": p999,
        "entropy": entropy,
        "otsu_threshold": otsu,
        "segmentation_threshold": segmentation_threshold,
        "threshold_area": area,
        "threshold_area_fraction": float(area / image.size),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_fill": bbox_fill,
        "threshold_centroid_x": threshold_centroid_x,
        "threshold_centroid_y": threshold_centroid_y,
        "nonzero_fraction": nonzero_fraction,
        "saturation_fraction": saturation_fraction,
        "bright_fraction_64": bright_fraction_64,
        "bright_fraction_128": bright_fraction_128,
        "bright_fraction_200": bright_fraction_200,
        "active_candidate": active_candidate,
    }

    result.update(geometry)
    return result


def process_video(video_path):
    output_path = OUTPUT_DIR / f"{video_path.stem}_features.csv"
    summary_path = OUTPUT_DIR / f"{video_path.stem}_summary.json"

    temporary_path = output_path.with_suffix(".csv.tmp")
    temporary_summary = summary_path.with_suffix(".json.tmp")

    print("Input:", video_path)
    print("Output:", output_path)

    with video_path.open("rb") as handle:
        header = handle.read(12)

        if header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise RuntimeError("Not a standard RIFF AVI file")

        movi_start, movi_end = find_movi(
            handle,
            video_path.stat().st_size,
        )

        frame_chunks = []
        collect_frame_chunks(
            handle,
            movi_start,
            movi_end,
            frame_chunks,
        )

        print("Frame chunks:", len(frame_chunks))

        previous = None
        area_window = deque(maxlen=ROLLING_WINDOW)
        intensity_window = deque(maxlen=ROLLING_WINDOW)

        rows = []
        active_frames = 0

        for frame_index, (offset, size) in enumerate(frame_chunks):
            image = decode_gray_frame(handle, offset, size)
            features = extract_features(image)

            if features["active_candidate"]:
                active_frames += 1

            area_window.append(features["threshold_area"])
            intensity_window.append(features["mean_intensity"])

            if previous is None:
                delta_area = 0.0
                delta_mean_intensity = 0.0
                delta_total_intensity = 0.0
                centroid_motion = 0.0
            else:
                delta_area = (
                    features["threshold_area"]
                    - previous["threshold_area"]
                )
                delta_mean_intensity = (
                    features["mean_intensity"]
                    - previous["mean_intensity"]
                )
                delta_total_intensity = (
                    features["total_intensity"]
                    - previous["total_intensity"]
                )

                coordinates = [
                    features["centroid_x"],
                    features["centroid_y"],
                    previous["centroid_x"],
                    previous["centroid_y"],
                ]

                if all(math.isfinite(value) for value in coordinates):
                    centroid_motion = math.hypot(
                        features["centroid_x"]
                        - previous["centroid_x"],
                        features["centroid_y"]
                        - previous["centroid_y"],
                    )
                else:
                    centroid_motion = 0.0

            row = {
                "video_id": video_path.stem,
                "frame_index": frame_index,
                "physical_time_s": frame_index / PHYSICAL_FPS,
                "physical_time_ms": 1000.0 * frame_index / PHYSICAL_FPS,
                **features,
                "delta_area": delta_area,
                "delta_mean_intensity": delta_mean_intensity,
                "delta_total_intensity": delta_total_intensity,
                "centroid_motion": centroid_motion,
                "rolling_area_mean": float(np.mean(area_window)),
                "rolling_area_std": float(np.std(area_window)),
                "rolling_intensity_mean": float(
                    np.mean(intensity_window)
                ),
                "rolling_intensity_std": float(
                    np.std(intensity_window)
                ),
            }

            rows.append(row)
            previous = features

            if (
                frame_index == 0
                or (frame_index + 1) % 100 == 0
                or frame_index + 1 == len(frame_chunks)
            ):
                print(
                    f"  frame {frame_index + 1}/{len(frame_chunks)}",
                    flush=True,
                )

    fieldnames = list(rows[0].keys())

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "video_id": video_path.stem,
        "video_path": str(video_path),
        "frame_count": len(rows),
        "physical_fps": PHYSICAL_FPS,
        "physical_duration_s": len(rows) / PHYSICAL_FPS,
        "active_candidate_frames": active_frames,
        "inactive_candidate_frames": len(rows) - active_frames,
        "active_candidate_fraction": (
            active_frames / len(rows) if rows else 0.0
        ),
        "mean_intensity_min": min(
            row["mean_intensity"] for row in rows
        ),
        "mean_intensity_max": max(
            row["mean_intensity"] for row in rows
        ),
        "threshold_area_min": min(
            row["threshold_area"] for row in rows
        ),
        "threshold_area_max": max(
            row["threshold_area"] for row in rows
        ),
    }

    with temporary_summary.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(summary, output, indent=2)

    temporary_path.replace(output_path)
    temporary_summary.replace(summary_path)

    print()
    print(json.dumps(summary, indent=2))
    print()
    print("Saved:", output_path)
    print("Saved:", summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="P01",
        help="Video identifier, for example P01",
    )
    arguments = parser.parse_args()

    video_path = VIDEO_DIR / f"RHF_MPM_{arguments.video}.avi"

    if not video_path.exists():
        print("Video was not found:", video_path)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_video(video_path)


if __name__ == "__main__":
    main()
