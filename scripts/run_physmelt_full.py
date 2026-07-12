#!/usr/bin/env python3

# Restrict numerical libraries before importing NumPy or scikit-learn.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
import importlib.util
import itertools
import json
import math
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path.home() / "PhysMelt"

VIDEO_DIR = ROOT / "data/raw/NIST_RHF/MPM_AVIs"
FEATURE_DIR = ROOT / "data/features/frame_features"
CACHE_DIR = ROOT / "data/model_cache"
MANIFEST_DIR = ROOT / "data/manifests"

EXTRACTOR = ROOT / "scripts/extract_frame_features.py"

EXPERIMENT_DIR = ROOT / "outputs/experiments"
TASK_DIR = EXPERIMENT_DIR / "tasks"
RESULT_DIR = EXPERIMENT_DIR / "results"
PREDICTION_DIR = EXPERIMENT_DIR / "predictions"
REPORT_DIR = ROOT / "outputs/reports"
FIGURE_DIR = ROOT / "outputs/figures"
LOG_DIR = ROOT / "logs/full_pipeline"

ALL_FEATURES = [
    "threshold_area",
    "threshold_area_fraction",
    "mean_intensity",
    "std_intensity",
    "total_intensity",
    "effective_area",
    "pixel_max",
    "p95",
    "p99",
    "p999",
    "entropy",
    "saturation_fraction",
    "bright_fraction_64",
    "bright_fraction_128",
    "bright_fraction_200",
    "bbox_width",
    "bbox_height",
    "bbox_fill",
    "centroid_x",
    "centroid_y",
    "major_sigma",
    "minor_sigma",
    "eccentricity",
    "orientation_deg",
    "tail_skewness",
    "abs_tail_skewness",
    "outer_intensity_fraction",
    "delta_area",
    "delta_mean_intensity",
    "delta_total_intensity",
    "centroid_motion",
    "rolling_area_mean",
    "rolling_area_std",
    "rolling_intensity_mean",
    "rolling_intensity_std",
]

FEATURE_SETS = {
    "basic": [
        "threshold_area",
        "mean_intensity",
    ],
    "appearance": [
        "threshold_area",
        "mean_intensity",
        "std_intensity",
        "total_intensity",
        "effective_area",
        "p99",
        "p999",
        "saturation_fraction",
        "bright_fraction_128",
        "bright_fraction_200",
    ],
    "physical": [
        "threshold_area",
        "mean_intensity",
        "std_intensity",
        "total_intensity",
        "effective_area",
        "p99",
        "p999",
        "saturation_fraction",
        "bright_fraction_128",
        "bright_fraction_200",
        "bbox_width",
        "bbox_height",
        "bbox_fill",
        "major_sigma",
        "minor_sigma",
        "eccentricity",
        "abs_tail_skewness",
        "outer_intensity_fraction",
        "centroid_motion",
    ],
    "physical_temporal": [
        "threshold_area",
        "mean_intensity",
        "std_intensity",
        "total_intensity",
        "effective_area",
        "p99",
        "p999",
        "saturation_fraction",
        "bright_fraction_128",
        "bright_fraction_200",
        "bbox_width",
        "bbox_height",
        "bbox_fill",
        "major_sigma",
        "minor_sigma",
        "eccentricity",
        "abs_tail_skewness",
        "outer_intensity_fraction",
        "delta_area",
        "delta_mean_intensity",
        "delta_total_intensity",
        "centroid_motion",
        "rolling_area_mean",
        "rolling_area_std",
        "rolling_intensity_mean",
        "rolling_intensity_std",
    ],
}

THERMAL_FEATURES = [
    "threshold_area",
    "mean_intensity",
    "bright_fraction_200",
    "outer_intensity_fraction",
]

THERMAL_WEIGHTS = np.array(
    [1.0, 1.0, 0.7, 0.3],
    dtype=np.float64,
)

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}


def ensure_directories():
    for directory in [
        FEATURE_DIR,
        CACHE_DIR,
        MANIFEST_DIR,
        TASK_DIR,
        RESULT_DIR,
        PREDICTION_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def atomic_json_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)

    temporary.replace(path)


def read_meminfo():
    values = {}

    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                number = raw_value.strip().split()[0]
                values[key] = int(number) * 1024
    except Exception:
        pass

    return values


def available_memory_gb():
    values = read_meminfo()
    value = values.get("MemAvailable", 0)
    return value / 1024**3


def swap_used_gb():
    values = read_meminfo()
    total = values.get("SwapTotal", 0)
    free = values.get("SwapFree", 0)
    return max(total - free, 0) / 1024**3


def short_last_line(path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""

            while position > 0 and buffer.count(b"\n") < 3:
                step = min(4096, position)
                position -= step
                handle.seek(position)
                buffer = handle.read(step) + buffer

        lines = [
            line.decode("utf-8", errors="replace").strip()
            for line in buffer.splitlines()
            if line.strip()
        ]

        return lines[-1] if lines else ""
    except Exception:
        return ""


def process_prefix():
    prefix = []

    if shutil.which("nice"):
        prefix += ["nice", "-n", "15"]

    if shutil.which("ionice"):
        prefix += ["ionice", "-c", "2", "-n", "7"]

    return prefix


def terminate_own_processes(running):
    for item in running.values():
        process = item["process"]

        try:
            os.killpg(process.pid, signal.SIGCONT)
        except Exception:
            pass

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass


def memory_aware_subprocess_scheduler(
    tasks,
    maximum_workers,
    reserve_gb,
    estimated_worker_gb,
    emergency_gb,
    resume_gb,
    stage_name,
):
    pending = list(tasks)
    running = {}
    completed = []
    failed = []
    paused = False
    last_status = 0.0

    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    child_environment["OMP_NUM_THREADS"] = "1"
    child_environment["OPENBLAS_NUM_THREADS"] = "1"
    child_environment["MKL_NUM_THREADS"] = "1"
    child_environment["NUMEXPR_NUM_THREADS"] = "1"

    print()
    print("=" * 78)
    print(stage_name)
    print("=" * 78)
    print("Pending jobs:", len(pending))
    print("Maximum workers:", maximum_workers)
    print("Reserved RAM:", f"{reserve_gb:.1f} GB")
    print("Emergency pause threshold:", f"{emergency_gb:.1f} GB")
    print("Resume threshold:", f"{resume_gb:.1f} GB")
    print()

    try:
        while pending or running:
            current_available = available_memory_gb()

            if current_available < emergency_gb and running and not paused:
                print(
                    f"[MEMORY] available={current_available:.1f} GB; "
                    "temporarily pausing PhysMelt workers"
                )

                for item in running.values():
                    try:
                        os.killpg(item["process"].pid, signal.SIGSTOP)
                    except Exception:
                        pass

                paused = True

            elif paused and current_available >= resume_gb:
                print(
                    f"[MEMORY] available={current_available:.1f} GB; "
                    "resuming PhysMelt workers"
                )

                for item in running.values():
                    try:
                        os.killpg(item["process"].pid, signal.SIGCONT)
                    except Exception:
                        pass

                paused = False

            finished_names = []

            for name, item in list(running.items()):
                return_code = item["process"].poll()

                if return_code is None:
                    continue

                item["log_handle"].close()
                finished_names.append(name)

                if return_code == 0 and item["output_check"]():
                    completed.append(name)
                    print(f"[DONE] {name}")
                else:
                    failed.append(name)
                    print(
                        f"[FAILED] {name}, return code={return_code}, "
                        f"log={item['log_path']}"
                    )

                    last_line = short_last_line(item["log_path"])
                    if last_line:
                        print("         last log line:", last_line)

            for name in finished_names:
                del running[name]

            if not paused:
                current_available = available_memory_gb()
                memory_slots = int(
                    max(
                        0.0,
                        current_available - reserve_gb,
                    )
                    // max(estimated_worker_gb, 0.1)
                )

                target_running = min(
                    maximum_workers,
                    max(memory_slots, 0),
                )

                while pending and len(running) < target_running:
                    task = pending.pop(0)
                    name = task["name"]
                    command = process_prefix() + task["command"]
                    log_path = Path(task["log_path"])
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                    log_handle = log_path.open(
                        "w",
                        encoding="utf-8",
                        buffering=1,
                    )

                    print(
                        f"[START] {name} | "
                        f"available={current_available:.1f} GB"
                    )

                    process = subprocess.Popen(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=child_environment,
                        text=True,
                        preexec_fn=os.setsid,
                    )

                    running[name] = {
                        "process": process,
                        "log_handle": log_handle,
                        "log_path": log_path,
                        "output_check": task["output_check"],
                    }

            now = time.time()

            if now - last_status >= 5.0:
                current_available = available_memory_gb()

                print(
                    f"[STATUS] done={len(completed)} "
                    f"failed={len(failed)} "
                    f"running={len(running)} "
                    f"pending={len(pending)} "
                    f"available_ram={current_available:.1f}GB "
                    f"swap_used={swap_used_gb():.1f}GB"
                )

                for name, item in list(running.items())[:4]:
                    last_line = short_last_line(item["log_path"])
                    if last_line:
                        print(f"         {name}: {last_line}")

                last_status = now

            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received. Stopping only PhysMelt child processes.")
        terminate_own_processes(running)

        for item in running.values():
            try:
                item["log_handle"].close()
            except Exception:
                pass

    print()
    print("Stage completed:", stage_name)
    print("Successful:", len(completed))
    print("Failed:", len(failed))

    if failed:
        print("Failed jobs:", ", ".join(failed))

    return failed


def extraction_complete(video_name):
    feature_path = FEATURE_DIR / f"RHF_MPM_{video_name}_features.csv"
    summary_path = FEATURE_DIR / f"RHF_MPM_{video_name}_summary.json"

    if not feature_path.exists() or not summary_path.exists():
        return False

    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        return (
            int(summary.get("frame_count", 0)) == 1498
            and feature_path.stat().st_size > 1000
        )
    except Exception:
        return False


def run_extraction(args):
    if not EXTRACTOR.exists():
        print("Missing extractor:", EXTRACTOR)
        return False

    videos = [
        f"P{number:02d}"
        for number in range(1, 56)
    ]

    tasks = []

    for video_name in videos:
        if extraction_complete(video_name):
            print("[SKIP]", video_name, "already extracted")
            continue

        feature_path = (
            FEATURE_DIR
            / f"RHF_MPM_{video_name}_features.csv"
        )

        summary_path = (
            FEATURE_DIR
            / f"RHF_MPM_{video_name}_summary.json"
        )

        tasks.append({
            "name": video_name,
            "command": [
                sys.executable,
                str(EXTRACTOR),
                "--video",
                video_name,
            ],
            "log_path": (
                LOG_DIR
                / "extraction"
                / f"extract_{video_name}.log"
            ),
            "output_check": (
                lambda f=feature_path, s=summary_path:
                f.exists()
                and s.exists()
                and extraction_complete(
                    s.stem
                    .replace("RHF_MPM_", "")
                    .replace("_summary", "")
                )
            ),
        })

    if not tasks:
        print("All 55 videos are already extracted.")
        return True

    failed = memory_aware_subprocess_scheduler(
        tasks=tasks,
        maximum_workers=args.extract_workers,
        reserve_gb=args.reserve_gb,
        estimated_worker_gb=0.75,
        emergency_gb=args.emergency_gb,
        resume_gb=args.resume_gb,
        stage_name="PHYS-MELT FEATURE EXTRACTION",
    )

    return len(failed) == 0


def parse_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except Exception:
        return np.nan


def make_partitions(video_names):
    shuffled = list(video_names)
    random.Random(42).shuffle(shuffled)

    split_lookup = {}
    fold_lookup = {}

    for position, video_name in enumerate(shuffled):
        if position < 33:
            split_code = 0
        elif position < 44:
            split_code = 1
        else:
            split_code = 2

        split_lookup[video_name] = split_code
        fold_lookup[video_name] = position % 5

    return split_lookup, fold_lookup


def cache_complete():
    required = [
        CACHE_DIR / "features.npy",
        CACHE_DIR / "active.npy",
        CACHE_DIR / "video_index.npy",
        CACHE_DIR / "frame_index.npy",
        CACHE_DIR / "physical_time_s.npy",
        CACHE_DIR / "split_code.npy",
        CACHE_DIR / "fold.npy",
        CACHE_DIR / "metadata.json",
    ]

    return all(path.exists() for path in required)


def build_cache():
    video_names = [
        f"RHF_MPM_P{number:02d}"
        for number in range(1, 56)
    ]

    missing = []

    for video_name in video_names:
        path = FEATURE_DIR / f"{video_name}_features.csv"

        if not path.exists():
            missing.append(video_name)

    if missing:
        print("Feature files are missing:")
        print(", ".join(missing))
        return False

    split_lookup, fold_lookup = make_partitions(video_names)

    rows_per_video = {}
    total_rows = 0

    for video_name in video_names:
        summary_path = FEATURE_DIR / f"{video_name}_summary.json"

        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)

        count = int(summary["frame_count"])
        rows_per_video[video_name] = count
        total_rows += count

    print("Videos:", len(video_names))
    print("Total frames:", total_rows)
    print("Features:", len(ALL_FEATURES))

    feature_matrix = np.empty(
        (total_rows, len(ALL_FEATURES)),
        dtype=np.float32,
    )
    active = np.empty(total_rows, dtype=np.bool_)
    video_index = np.empty(total_rows, dtype=np.int16)
    frame_index = np.empty(total_rows, dtype=np.int32)
    physical_time_s = np.empty(total_rows, dtype=np.float32)
    split_code = np.empty(total_rows, dtype=np.int8)
    fold_array = np.empty(total_rows, dtype=np.int8)

    combined_path = FEATURE_DIR / "all_frame_features.csv"
    combined_tmp = combined_path.with_suffix(".csv.tmp")

    output_handle = combined_tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    combined_writer = None
    offset = 0

    try:
        for current_video_index, video_name in enumerate(video_names):
            path = FEATURE_DIR / f"{video_name}_features.csv"

            print(
                f"[{current_video_index + 1:02d}/55] "
                f"loading {path.name}",
                flush=True,
            )

            with path.open(
                newline="",
                encoding="utf-8",
            ) as input_handle:

                reader = csv.DictReader(input_handle)

                if reader.fieldnames is None:
                    raise RuntimeError(
                        f"Missing CSV header: {path}"
                    )

                missing_columns = [
                    name
                    for name in ALL_FEATURES
                    if name not in reader.fieldnames
                ]

                if missing_columns:
                    raise RuntimeError(
                        f"Missing columns in {path}: "
                        + ", ".join(missing_columns)
                    )

                if combined_writer is None:
                    combined_fields = list(reader.fieldnames) + [
                        "split",
                        "fold",
                    ]

                    combined_writer = csv.DictWriter(
                        output_handle,
                        fieldnames=combined_fields,
                    )
                    combined_writer.writeheader()

                row_count = 0

                for row in reader:
                    target = offset + row_count

                    feature_matrix[target, :] = [
                        parse_float(row[name])
                        for name in ALL_FEATURES
                    ]

                    active[target] = (
                        str(row.get("active_candidate", "0")) == "1"
                    )
                    video_index[target] = current_video_index
                    frame_index[target] = int(row["frame_index"])
                    physical_time_s[target] = float(
                        row["physical_time_s"]
                    )
                    split_code[target] = split_lookup[video_name]
                    fold_array[target] = fold_lookup[video_name]

                    output_row = dict(row)
                    output_row["split"] = SPLIT_NAMES[
                        split_lookup[video_name]
                    ]
                    output_row["fold"] = fold_lookup[video_name]
                    combined_writer.writerow(output_row)

                    row_count += 1

                expected = rows_per_video[video_name]

                if row_count != expected:
                    raise RuntimeError(
                        f"{video_name}: read {row_count}, "
                        f"expected {expected}"
                    )

                offset += row_count

    finally:
        output_handle.close()

    if offset != total_rows:
        print(
            "Unexpected total row count:",
            offset,
            "expected:",
            total_rows,
        )
        return False

    combined_tmp.replace(combined_path)

    np.save(CACHE_DIR / "features.npy", feature_matrix)
    np.save(CACHE_DIR / "active.npy", active)
    np.save(CACHE_DIR / "video_index.npy", video_index)
    np.save(CACHE_DIR / "frame_index.npy", frame_index)
    np.save(
        CACHE_DIR / "physical_time_s.npy",
        physical_time_s,
    )
    np.save(CACHE_DIR / "split_code.npy", split_code)
    np.save(CACHE_DIR / "fold.npy", fold_array)

    partition_path = MANIFEST_DIR / "video_partitions.csv"

    with partition_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "video_index",
                "split",
                "fold",
            ],
        )
        writer.writeheader()

        for index, video_name in enumerate(video_names):
            writer.writerow({
                "video_id": video_name,
                "video_index": index,
                "split": SPLIT_NAMES[
                    split_lookup[video_name]
                ],
                "fold": fold_lookup[video_name],
            })

    metadata = {
        "video_names": video_names,
        "feature_names": ALL_FEATURES,
        "feature_sets": FEATURE_SETS,
        "thermal_features": THERMAL_FEATURES,
        "total_frames": total_rows,
        "active_frames": int(active.sum()),
        "inactive_frames": int((~active).sum()),
        "active_fraction": float(active.mean()),
        "physical_fps": 20000.0,
        "split_video_counts": {
            split_name: sum(
                SPLIT_NAMES[
                    split_lookup[video_name]
                ] == split_name
                for video_name in video_names
            )
            for split_name in [
                "train",
                "validation",
                "test",
            ]
        },
        "fold_video_counts": {
            str(fold): sum(
                fold_lookup[video_name] == fold
                for video_name in video_names
            )
            for fold in range(5)
        },
        "combined_csv": str(combined_path),
        "partition_file": str(partition_path),
    }

    atomic_json_write(
        CACHE_DIR / "metadata.json",
        metadata,
    )

    print()
    print(json.dumps(metadata, indent=2))
    print()
    print("Model cache saved to:", CACHE_DIR)

    return True


def require_sklearn():
    return importlib.util.find_spec("sklearn") is not None


def load_cache():
    with (
        CACHE_DIR / "metadata.json"
    ).open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        "metadata": metadata,
        "features": np.load(
            CACHE_DIR / "features.npy",
            mmap_mode="r",
        ),
        "active": np.load(
            CACHE_DIR / "active.npy",
            mmap_mode="r",
        ),
        "video_index": np.load(
            CACHE_DIR / "video_index.npy",
            mmap_mode="r",
        ),
        "frame_index": np.load(
            CACHE_DIR / "frame_index.npy",
            mmap_mode="r",
        ),
        "split_code": np.load(
            CACHE_DIR / "split_code.npy",
            mmap_mode="r",
        ),
        "fold": np.load(
            CACHE_DIR / "fold.npy",
            mmap_mode="r",
        ),
    }


def finite_imputer_fit(values):
    clean = np.where(np.isfinite(values), values, np.nan)
    medians = np.nanmedian(clean, axis=0)
    medians = np.where(
        np.isfinite(medians),
        medians,
        0.0,
    )
    return medians


def finite_imputer_transform(values, medians):
    result = np.asarray(values, dtype=np.float64).copy()
    invalid = ~np.isfinite(result)

    if invalid.any():
        row_indices, column_indices = np.where(invalid)
        result[row_indices, column_indices] = medians[
            column_indices
        ]

    return result


def standardizer_fit(values):
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    standard_deviations = np.where(
        standard_deviations < 1e-9,
        1.0,
        standard_deviations,
    )

    return means, standard_deviations


def standardizer_transform(
    values,
    means,
    standard_deviations,
):
    return (values - means) / standard_deviations


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(axis=1, keepdims=True)
    denominator = np.maximum(denominator, 1e-12)

    return exponentials / denominator


def thermal_score_fit_transform(
    matrix,
    feature_names,
    train_indices,
    test_indices,
):
    indices = [
        feature_names.index(name)
        for name in THERMAL_FEATURES
    ]

    train_values = np.asarray(
        matrix[train_indices][:, indices],
        dtype=np.float64,
    )
    test_values = np.asarray(
        matrix[test_indices][:, indices],
        dtype=np.float64,
    )

    medians = finite_imputer_fit(train_values)
    train_values = finite_imputer_transform(
        train_values,
        medians,
    )
    test_values = finite_imputer_transform(
        test_values,
        medians,
    )

    means, standard_deviations = standardizer_fit(
        train_values
    )

    train_standardized = standardizer_transform(
        train_values,
        means,
        standard_deviations,
    )
    test_standardized = standardizer_transform(
        test_values,
        means,
        standard_deviations,
    )

    return (
        train_standardized @ THERMAL_WEIGHTS,
        test_standardized @ THERMAL_WEIGHTS,
    )


def reorder_components(
    train_labels,
    test_labels,
    train_probabilities,
    test_probabilities,
    train_thermal_score,
):
    cluster_scores = []

    for cluster in range(3):
        mask = train_labels == cluster

        if mask.any():
            score = float(
                train_thermal_score[mask].mean()
            )
        else:
            score = float("inf")

        cluster_scores.append(score)

    old_components_in_new_order = np.argsort(
        cluster_scores
    )

    old_to_new = np.empty(3, dtype=np.int64)

    for new_label, old_label in enumerate(
        old_components_in_new_order
    ):
        old_to_new[old_label] = new_label

    reordered_train_labels = old_to_new[train_labels]
    reordered_test_labels = old_to_new[test_labels]

    reordered_train_probabilities = (
        train_probabilities[
            :,
            old_components_in_new_order,
        ]
    )
    reordered_test_probabilities = (
        test_probabilities[
            :,
            old_components_in_new_order,
        ]
    )

    return (
        reordered_train_labels,
        reordered_test_labels,
        reordered_train_probabilities,
        reordered_test_probabilities,
        old_components_in_new_order,
    )


def anchored_means_initialization(
    train_features,
    train_thermal_score,
):
    order = np.argsort(train_thermal_score)
    count = len(order)
    window = max(32, int(round(count * 0.05)))

    centers = []

    for quantile in [0.10, 0.50, 0.90]:
        center = int(round((count - 1) * quantile))
        start = max(0, center - window // 2)
        end = min(count, start + window)

        selected = order[start:end]

        centers.append(
            train_features[selected].mean(axis=0)
        )

    return np.vstack(centers)


def viterbi_segment(probabilities):
    transition = np.array([
        [0.985, 0.014, 0.001],
        [0.007, 0.986, 0.007],
        [0.001, 0.014, 0.985],
    ], dtype=np.float64)

    initial = np.array(
        [0.20, 0.60, 0.20],
        dtype=np.float64,
    )

    log_transition = np.log(
        np.maximum(transition, 1e-12)
    )
    log_initial = np.log(
        np.maximum(initial, 1e-12)
    )
    log_emission = np.log(
        np.maximum(probabilities, 1e-12)
    )

    length = len(probabilities)

    if length == 0:
        return np.empty(0, dtype=np.int8)

    delta = np.empty((length, 3), dtype=np.float64)
    pointer = np.empty((length, 3), dtype=np.int8)

    delta[0] = log_initial + log_emission[0]
    pointer[0] = 0

    for time_index in range(1, length):
        candidates = (
            delta[time_index - 1][:, None]
            + log_transition
        )

        pointer[time_index] = np.argmax(
            candidates,
            axis=0,
        )

        delta[time_index] = (
            candidates[
                pointer[time_index],
                np.arange(3),
            ]
            + log_emission[time_index]
        )

    states = np.empty(length, dtype=np.int8)
    states[-1] = int(np.argmax(delta[-1]))

    for time_index in range(length - 2, -1, -1):
        states[time_index] = pointer[
            time_index + 1,
            states[time_index + 1],
        ]

    return states


def temporal_decode(
    probabilities,
    video_indices,
    frame_indices,
):
    output = np.empty(
        len(probabilities),
        dtype=np.int8,
    )

    for video in np.unique(video_indices):
        positions = np.flatnonzero(
            video_indices == video
        )

        positions = positions[
            np.argsort(frame_indices[positions])
        ]

        sorted_frames = frame_indices[positions]

        if len(positions) == 0:
            continue

        boundaries = np.flatnonzero(
            np.diff(sorted_frames) > 1
        ) + 1

        segments = np.split(positions, boundaries)

        for segment in segments:
            output[segment] = viterbi_segment(
                probabilities[segment]
            )

    return output


def sequence_metrics(
    labels,
    video_indices,
    frame_indices,
):
    total_transitions = 0
    total_possible = 0
    short_run_frames = 0
    total_frames = 0
    run_lengths = []

    for video in np.unique(video_indices):
        positions = np.flatnonzero(
            video_indices == video
        )
        positions = positions[
            np.argsort(frame_indices[positions])
        ]

        sorted_frames = frame_indices[positions]
        boundaries = np.flatnonzero(
            np.diff(sorted_frames) > 1
        ) + 1

        for segment in np.split(positions, boundaries):
            if len(segment) == 0:
                continue

            segment_labels = labels[segment]
            total_frames += len(segment_labels)

            if len(segment_labels) > 1:
                total_transitions += int(
                    np.sum(
                        segment_labels[1:]
                        != segment_labels[:-1]
                    )
                )
                total_possible += len(segment_labels) - 1

            start = 0

            for position in range(
                1,
                len(segment_labels) + 1,
            ):
                is_boundary = (
                    position == len(segment_labels)
                    or segment_labels[position]
                    != segment_labels[position - 1]
                )

                if is_boundary:
                    run_length = position - start
                    run_lengths.append(run_length)

                    if run_length <= 2:
                        short_run_frames += run_length

                    start = position

    switches_per_1000 = (
        1000.0 * total_transitions
        / max(total_possible, 1)
    )

    return {
        "switches": total_transitions,
        "switches_per_1000": switches_per_1000,
        "short_run_frame_fraction": (
            short_run_frames
            / max(total_frames, 1)
        ),
        "median_run_length": (
            float(np.median(run_lengths))
            if run_lengths
            else 0.0
        ),
        "mean_run_length": (
            float(np.mean(run_lengths))
            if run_lengths
            else 0.0
        ),
    }


def cluster_metrics(
    features,
    labels,
    probabilities,
    thermal_score,
    seed,
):
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    result = {}

    unique_labels = np.unique(labels)

    if len(unique_labels) >= 2:
        sample_count = min(8000, len(features))
        random_generator = np.random.default_rng(seed)

        if sample_count < len(features):
            sample_indices = random_generator.choice(
                len(features),
                size=sample_count,
                replace=False,
            )
        else:
            sample_indices = np.arange(len(features))

        sampled_features = features[sample_indices]
        sampled_labels = labels[sample_indices]

        try:
            result["silhouette"] = float(
                silhouette_score(
                    sampled_features,
                    sampled_labels,
                    metric="euclidean",
                )
            )
        except Exception:
            result["silhouette"] = float("nan")

        try:
            result["davies_bouldin"] = float(
                davies_bouldin_score(
                    sampled_features,
                    sampled_labels,
                )
            )
        except Exception:
            result["davies_bouldin"] = float("nan")

        try:
            result["calinski_harabasz"] = float(
                calinski_harabasz_score(
                    sampled_features,
                    sampled_labels,
                )
            )
        except Exception:
            result["calinski_harabasz"] = float("nan")

    else:
        result["silhouette"] = float("nan")
        result["davies_bouldin"] = float("nan")
        result["calinski_harabasz"] = float("nan")

    confidence = probabilities.max(axis=1)

    result["confidence_mean"] = float(
        confidence.mean()
    )
    result["confident_fraction_080"] = float(
        np.mean(confidence >= 0.80)
    )
    result["confident_fraction_095"] = float(
        np.mean(confidence >= 0.95)
    )

    thermal_means = []

    for cluster in range(3):
        mask = labels == cluster

        thermal_means.append(
            float(thermal_score[mask].mean())
            if mask.any()
            else float("nan")
        )

        result[f"cluster_{cluster}_fraction"] = float(
            mask.mean()
        )

    result["thermal_mean_under"] = thermal_means[0]
    result["thermal_mean_nominal"] = thermal_means[1]
    result["thermal_mean_over"] = thermal_means[2]

    monotonic = (
        np.isfinite(thermal_means).all()
        and thermal_means[0]
        < thermal_means[1]
        < thermal_means[2]
    )

    result["thermal_monotonic"] = bool(monotonic)

    score_standard_deviation = float(
        np.std(thermal_score)
    )

    result["extreme_separation_std"] = (
        float(
            (
                thermal_means[2]
                - thermal_means[0]
            )
            / score_standard_deviation
        )
        if score_standard_deviation > 1e-9
        else float("nan")
    )

    return result


def edge_distillation(
    train_features,
    test_features,
    train_targets,
    test_targets,
    seed,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    model = LogisticRegression(
        max_iter=800,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
    )

    model.fit(train_features, train_targets)

    predictions = model.predict(test_features)

    sample = test_features[
        :min(10000, len(test_features))
    ]

    repetitions = 5
    start = time.perf_counter()

    for _ in range(repetitions):
        model.predict_proba(sample)

    elapsed = time.perf_counter() - start

    latency_microseconds = (
        1e6
        * elapsed
        / max(len(sample) * repetitions, 1)
    )

    return {
        "edge_accuracy": float(
            accuracy_score(test_targets, predictions)
        ),
        "edge_balanced_accuracy": float(
            balanced_accuracy_score(
                test_targets,
                predictions,
            )
        ),
        "edge_macro_f1": float(
            f1_score(
                test_targets,
                predictions,
                average="macro",
            )
        ),
        "edge_latency_us_per_frame": float(
            latency_microseconds
        ),
        "edge_parameter_count": int(
            model.coef_.size
            + model.intercept_.size
        ),
    }


def run_single_experiment(task_path):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    with Path(task_path).open(
        encoding="utf-8"
    ) as handle:
        task = json.load(handle)

    cache = load_cache()

    metadata = cache["metadata"]
    feature_names = metadata["feature_names"]
    matrix = cache["features"]
    active = np.asarray(cache["active"])
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])
    split_code = np.asarray(cache["split_code"])
    folds = np.asarray(cache["fold"])

    feature_set_name = task["feature_set"]
    model_name = task["model"]
    seed = int(task["seed"])
    protocol = task["protocol"]
    held_out_fold = task.get("fold")

    selected_feature_names = FEATURE_SETS[
        feature_set_name
    ]
    selected_columns = [
        feature_names.index(name)
        for name in selected_feature_names
    ]

    if protocol == "holdout":
        train_mask = active & (
            (split_code == 0)
            | (split_code == 1)
        )
        test_mask = active & (split_code == 2)

    elif protocol == "cross_validation":
        train_mask = active & (
            folds != int(held_out_fold)
        )
        test_mask = active & (
            folds == int(held_out_fold)
        )

    else:
        raise ValueError(
            f"Unknown protocol: {protocol}"
        )

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    train_features = np.asarray(
        matrix[train_indices][:, selected_columns],
        dtype=np.float64,
    )
    test_features = np.asarray(
        matrix[test_indices][:, selected_columns],
        dtype=np.float64,
    )

    medians = finite_imputer_fit(train_features)

    train_features = finite_imputer_transform(
        train_features,
        medians,
    )
    test_features = finite_imputer_transform(
        test_features,
        medians,
    )

    means, standard_deviations = standardizer_fit(
        train_features
    )

    train_standardized = standardizer_transform(
        train_features,
        means,
        standard_deviations,
    )
    test_standardized = standardizer_transform(
        test_features,
        means,
        standard_deviations,
    )

    (
        train_thermal_score,
        test_thermal_score,
    ) = thermal_score_fit_transform(
        matrix,
        feature_names,
        train_indices,
        test_indices,
    )

    fit_start = time.perf_counter()

    if model_name == "kmeans":
        model = KMeans(
            n_clusters=3,
            n_init=20,
            max_iter=500,
            random_state=seed,
        )

        train_labels_original = model.fit_predict(
            train_standardized
        )
        test_labels_original = model.predict(
            test_standardized
        )

        train_distances = model.transform(
            train_standardized
        ) ** 2
        test_distances = model.transform(
            test_standardized
        ) ** 2

        temperature = float(
            np.median(
                np.min(
                    train_distances,
                    axis=1,
                )
            )
        )
        temperature = max(temperature, 1e-3)

        train_probabilities_original = softmax(
            -train_distances
            / (2.0 * temperature)
        )
        test_probabilities_original = softmax(
            -test_distances
            / (2.0 * temperature)
        )

    elif model_name in {
        "gmm",
        "anchored_gmm",
    }:
        keyword_arguments = {
            "n_components": 3,
            "covariance_type": "diag",
            "reg_covar": 1e-5,
            "max_iter": 500,
            "n_init": 5,
            "random_state": seed,
        }

        if model_name == "anchored_gmm":
            keyword_arguments["means_init"] = (
                anchored_means_initialization(
                    train_standardized,
                    train_thermal_score,
                )
            )
            keyword_arguments["weights_init"] = (
                np.array(
                    [0.25, 0.50, 0.25],
                    dtype=np.float64,
                )
            )
            keyword_arguments["n_init"] = 1

        model = GaussianMixture(
            **keyword_arguments
        )

        model.fit(train_standardized)

        train_labels_original = model.predict(
            train_standardized
        )
        test_labels_original = model.predict(
            test_standardized
        )

        train_probabilities_original = (
            model.predict_proba(
                train_standardized
            )
        )
        test_probabilities_original = (
            model.predict_proba(
                test_standardized
            )
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    fit_seconds = time.perf_counter() - fit_start

    (
        train_labels,
        test_labels,
        train_probabilities,
        test_probabilities,
        component_order,
    ) = reorder_components(
        train_labels_original,
        test_labels_original,
        train_probabilities_original,
        test_probabilities_original,
        train_thermal_score,
    )

    train_temporal_labels = temporal_decode(
        train_probabilities,
        video_index[train_indices],
        frame_index[train_indices],
    )
    test_temporal_labels = temporal_decode(
        test_probabilities,
        video_index[test_indices],
        frame_index[test_indices],
    )

    metrics = cluster_metrics(
        test_standardized,
        test_labels,
        test_probabilities,
        test_thermal_score,
        seed,
    )

    raw_sequence_metrics = sequence_metrics(
        test_labels,
        video_index[test_indices],
        frame_index[test_indices],
    )

    temporal_sequence_metrics = sequence_metrics(
        test_temporal_labels,
        video_index[test_indices],
        frame_index[test_indices],
    )

    for key, value in raw_sequence_metrics.items():
        metrics[f"raw_{key}"] = value

    for key, value in (
        temporal_sequence_metrics.items()
    ):
        metrics[f"temporal_{key}"] = value

    metrics["switch_reduction_fraction"] = (
        1.0
        - temporal_sequence_metrics[
            "switches_per_1000"
        ]
        / max(
            raw_sequence_metrics[
                "switches_per_1000"
            ],
            1e-12,
        )
    )

    if (
        model_name == "anchored_gmm"
        and feature_set_name
        == "physical_temporal"
        and protocol == "holdout"
    ):
        metrics.update(
            edge_distillation(
                train_standardized,
                test_standardized,
                train_temporal_labels,
                test_temporal_labels,
                seed,
            )
        )

    prediction_path = (
        PREDICTION_DIR
        / f"{task['task_id']}.npz"
    )

    np.savez(
        prediction_path,
        row_index=test_indices.astype(np.int64),
        video_index=video_index[
            test_indices
        ].astype(np.int16),
        frame_index=frame_index[
            test_indices
        ].astype(np.int32),
        raw_label=test_labels.astype(np.int8),
        temporal_label=test_temporal_labels.astype(
            np.int8
        ),
        confidence=test_probabilities.max(
            axis=1
        ).astype(np.float32),
        probabilities=test_probabilities.astype(
            np.float32
        ),
    )

    result = {
        "task_id": task["task_id"],
        "protocol": protocol,
        "fold": held_out_fold,
        "feature_set": feature_set_name,
        "model": model_name,
        "seed": seed,
        "train_frames": int(len(train_indices)),
        "test_frames": int(len(test_indices)),
        "feature_count": len(selected_columns),
        "fit_seconds": float(fit_seconds),
        "component_order": [
            int(value)
            for value in component_order
        ],
        "prediction_file": str(prediction_path),
        **metrics,
    }

    result_path = (
        RESULT_DIR
        / f"{task['task_id']}.json"
    )

    atomic_json_write(result_path, result)

    print(json.dumps(result, indent=2))


def result_complete(task_id):
    path = RESULT_DIR / f"{task_id}.json"

    if not path.exists():
        return False

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        return payload.get("task_id") == task_id
    except Exception:
        return False


def create_experiment_tasks(quick=False):
    tasks = []

    seeds = [42] if quick else [42, 43, 44]

    feature_sets = (
        ["basic", "physical_temporal"]
        if quick
        else list(FEATURE_SETS)
    )

    models = [
        "kmeans",
        "gmm",
        "anchored_gmm",
    ]

    for feature_set, model, seed in itertools.product(
        feature_sets,
        models,
        seeds,
    ):
        task_id = (
            f"holdout__{feature_set}"
            f"__{model}__seed{seed}"
        )

        tasks.append({
            "task_id": task_id,
            "protocol": "holdout",
            "fold": None,
            "feature_set": feature_set,
            "model": model,
            "seed": seed,
        })

    if not quick:
        for fold, seed in itertools.product(
            range(5),
            seeds,
        ):
            task_id = (
                "cv__physical_temporal"
                f"__anchored_gmm"
                f"__fold{fold}"
                f"__seed{seed}"
            )

            tasks.append({
                "task_id": task_id,
                "protocol": "cross_validation",
                "fold": fold,
                "feature_set": "physical_temporal",
                "model": "anchored_gmm",
                "seed": seed,
            })

    return tasks


def run_experiments(args):
    if not cache_complete():
        print(
            "Model cache is missing. "
            "Run the cache stage first."
        )
        return False

    if not require_sklearn():
        print()
        print("scikit-learn is not installed in this Python.")
        print("No package was installed or changed.")
        print(
            "Extraction and cache remain valid, "
            "but experiments cannot start."
        )
        return False

    task_payloads = create_experiment_tasks(
        quick=args.quick
    )

    scheduler_tasks = []

    for payload in task_payloads:
        task_id = payload["task_id"]

        if result_complete(task_id):
            print("[SKIP]", task_id)
            continue

        task_path = TASK_DIR / f"{task_id}.json"
        atomic_json_write(task_path, payload)

        result_path = RESULT_DIR / f"{task_id}.json"

        scheduler_tasks.append({
            "name": task_id,
            "command": [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-experiment",
                str(task_path),
            ],
            "log_path": (
                LOG_DIR
                / "experiments"
                / f"{task_id}.log"
            ),
            "output_check": (
                lambda p=result_path:
                p.exists()
                and p.stat().st_size > 100
            ),
        })

    if not scheduler_tasks:
        print("All experiment tasks are complete.")
        return True

    failed = memory_aware_subprocess_scheduler(
        tasks=scheduler_tasks,
        maximum_workers=args.experiment_workers,
        reserve_gb=args.reserve_gb,
        estimated_worker_gb=2.0,
        emergency_gb=args.emergency_gb,
        resume_gb=args.resume_gb,
        stage_name="PHYS-MELT EXPERIMENTS",
    )

    return len(failed) == 0


def mean_std(values):
    clean = np.array(
        [
            float(value)
            for value in values
            if value is not None
            and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )

    if len(clean) == 0:
        return float("nan"), float("nan")

    return (
        float(clean.mean()),
        float(clean.std(ddof=0)),
    )


def aggregate_results():
    from sklearn.metrics import adjusted_rand_score

    result_paths = sorted(
        RESULT_DIR.glob("*.json")
    )

    results = []

    for path in result_paths:
        try:
            with path.open(encoding="utf-8") as handle:
                results.append(json.load(handle))
        except Exception:
            print("Could not read:", path)

    if not results:
        print("No experiment results were found.")
        return False

    all_fields = sorted({
        field
        for result in results
        for field in result.keys()
        if field not in {
            "component_order",
            "prediction_file",
        }
    })

    raw_csv_path = (
        REPORT_DIR
        / "all_experiment_results.csv"
    )

    with raw_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=all_fields,
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result.get(field, "")
                for field in all_fields
            })

    stability_rows = []

    grouping = {}

    for result in results:
        key = (
            result["protocol"],
            result.get("fold"),
            result["feature_set"],
            result["model"],
        )
        grouping.setdefault(key, []).append(result)

    for key, group in grouping.items():
        raw_scores = []
        temporal_scores = []

        for first, second in itertools.combinations(
            group,
            2,
        ):
            first_prediction = np.load(
                first["prediction_file"]
            )
            second_prediction = np.load(
                second["prediction_file"]
            )

            first_rows = first_prediction["row_index"]
            second_rows = second_prediction["row_index"]

            if not np.array_equal(
                first_rows,
                second_rows,
            ):
                continue

            raw_scores.append(
                adjusted_rand_score(
                    first_prediction["raw_label"],
                    second_prediction["raw_label"],
                )
            )

            temporal_scores.append(
                adjusted_rand_score(
                    first_prediction[
                        "temporal_label"
                    ],
                    second_prediction[
                        "temporal_label"
                    ],
                )
            )

        stability_rows.append({
            "protocol": key[0],
            "fold": key[1],
            "feature_set": key[2],
            "model": key[3],
            "raw_seed_ari": (
                float(np.mean(raw_scores))
                if raw_scores
                else float("nan")
            ),
            "temporal_seed_ari": (
                float(np.mean(temporal_scores))
                if temporal_scores
                else float("nan")
            ),
            "pair_count": len(raw_scores),
        })

    stability_csv = (
        REPORT_DIR
        / "seed_stability.csv"
    )

    with stability_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol",
                "fold",
                "feature_set",
                "model",
                "raw_seed_ari",
                "temporal_seed_ari",
                "pair_count",
            ],
        )
        writer.writeheader()
        writer.writerows(stability_rows)

    summary_groups = {}

    for result in results:
        key = (
            result["protocol"],
            result["feature_set"],
            result["model"],
        )
        summary_groups.setdefault(key, []).append(result)

    summary_rows = []

    metric_names = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "confidence_mean",
        "confident_fraction_080",
        "extreme_separation_std",
        "raw_switches_per_1000",
        "temporal_switches_per_1000",
        "switch_reduction_fraction",
        "temporal_short_run_frame_fraction",
        "fit_seconds",
        "edge_accuracy",
        "edge_macro_f1",
        "edge_latency_us_per_frame",
    ]

    for key, group in sorted(summary_groups.items()):
        row = {
            "protocol": key[0],
            "feature_set": key[1],
            "model": key[2],
            "runs": len(group),
        }

        for metric in metric_names:
            mean_value, std_value = mean_std([
                item.get(metric)
                for item in group
            ])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value

        summary_rows.append(row)

    summary_fields = sorted({
        field
        for row in summary_rows
        for field in row.keys()
    })

    summary_csv = (
        REPORT_DIR
        / "experiment_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_path = (
        REPORT_DIR
        / "paper_tables.md"
    )

    holdout_rows = [
        row
        for row in summary_rows
        if row["protocol"] == "holdout"
    ]

    holdout_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["model"],
        )
    )

    with markdown_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("# PhysMelt experiment summary\n\n")
        handle.write(
            "| Features | Model | Silhouette | DB ↓ | "
            "Confidence | Separation | Switch reduction |\n"
        )
        handle.write(
            "|---|---|---:|---:|---:|---:|---:|\n"
        )

        for row in holdout_rows:
            handle.write(
                f"| {row['feature_set']} "
                f"| {row['model']} "
                f"| {row['silhouette_mean']:.3f} "
                f"| {row['davies_bouldin_mean']:.3f} "
                f"| {row['confidence_mean_mean']:.3f} "
                f"| {row['extreme_separation_std_mean']:.3f} "
                f"| {100.0 * row['switch_reduction_fraction_mean']:.1f}% |\n"
            )

        full_rows = [
            row
            for row in summary_rows
            if (
                row["protocol"] == "cross_validation"
                and row["feature_set"]
                == "physical_temporal"
                and row["model"]
                == "anchored_gmm"
            )
        ]

        if full_rows:
            row = full_rows[0]

            handle.write("\n## Five-fold full method\n\n")
            handle.write(
                f"- Silhouette: "
                f"{row['silhouette_mean']:.3f} "
                f"$\\pm$ {row['silhouette_std']:.3f}\n"
            )
            handle.write(
                f"- Extreme separation: "
                f"{row['extreme_separation_std_mean']:.3f} "
                f"$\\pm$ "
                f"{row['extreme_separation_std_std']:.3f}\n"
            )
            handle.write(
                f"- Temporal switch reduction: "
                f"{100.0 * row['switch_reduction_fraction_mean']:.1f}%\n"
            )

    make_figures(summary_rows)

    print()
    print("Raw results:", raw_csv_path)
    print("Summary:", summary_csv)
    print("Stability:", stability_csv)
    print("Paper table:", markdown_path)

    return True


def make_figures(summary_rows):
    if importlib.util.find_spec("matplotlib") is None:
        print(
            "matplotlib is unavailable; "
            "tables were generated without plots."
        )
        return

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    holdout = [
        row
        for row in summary_rows
        if (
            row["protocol"] == "holdout"
            and row["feature_set"]
            == "physical_temporal"
        )
    ]

    if holdout:
        labels = [
            row["model"]
            for row in holdout
        ]
        silhouettes = [
            row["silhouette_mean"]
            for row in holdout
        ]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, silhouettes)
        axis.set_ylabel("Silhouette score")
        axis.set_xlabel("Model")
        axis.set_title(
            "Physical-temporal representation"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR
            / "01_model_silhouette.pdf",
            bbox_inches="tight",
        )
        figure.savefig(
            FIGURE_DIR
            / "01_model_silhouette.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)

        raw_switches = [
            row["raw_switches_per_1000_mean"]
            for row in holdout
        ]
        temporal_switches = [
            row[
                "temporal_switches_per_1000_mean"
            ]
            for row in holdout
        ]

        x_positions = np.arange(len(labels))
        width = 0.36

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(
            x_positions - width / 2,
            raw_switches,
            width,
            label="Frame-level",
        )
        axis.bar(
            x_positions + width / 2,
            temporal_switches,
            width,
            label="Temporal",
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels)
        axis.set_ylabel(
            "State switches per 1,000 frames"
        )
        axis.set_title(
            "Temporal stabilization"
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR
            / "02_temporal_switches.pdf",
            bbox_inches="tight",
        )
        figure.savefig(
            FIGURE_DIR
            / "02_temporal_switches.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)

    ablation = [
        row
        for row in summary_rows
        if (
            row["protocol"] == "holdout"
            and row["model"]
            == "anchored_gmm"
        )
    ]

    if ablation:
        labels = [
            row["feature_set"]
            for row in ablation
        ]
        separation = [
            row[
                "extreme_separation_std_mean"
            ]
            for row in ablation
        ]
        confidence = [
            row["confidence_mean_mean"]
            for row in ablation
        ]

        figure, axis = plt.subplots(figsize=(10, 5))
        x_positions = np.arange(len(labels))

        axis.plot(
            x_positions,
            separation,
            marker="o",
            label="Extreme separation",
        )
        axis.plot(
            x_positions,
            confidence,
            marker="s",
            label="Mean confidence",
        )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            labels,
            rotation=15,
            ha="right",
        )
        axis.set_title(
            "Feature ablation for anchored GMM"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR
            / "03_feature_ablation.pdf",
            bbox_inches="tight",
        )
        figure.savefig(
            FIGURE_DIR
            / "03_feature_ablation.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def preflight():
    print("=== PhysMelt preflight ===")
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])
    print("CPU count:", os.cpu_count())
    print(
        "Available RAM:",
        f"{available_memory_gb():.1f} GB",
    )
    print(
        "Swap used:",
        f"{swap_used_gb():.1f} GB",
    )
    print("GPU disabled for children: yes")
    print("Video directory:", VIDEO_DIR)
    print(
        "AVI count:",
        len(list(VIDEO_DIR.glob("*.avi"))),
    )
    print("Extractor:", EXTRACTOR)
    print("Extractor exists:", EXTRACTOR.exists())
    print("NumPy:", np.__version__)
    print(
        "scikit-learn:",
        "FOUND"
        if require_sklearn()
        else "NOT FOUND",
    )
    print(
        "matplotlib:",
        "FOUND"
        if importlib.util.find_spec(
            "matplotlib"
        )
        else "NOT FOUND",
    )
    print("Existing completed videos:", sum(
        extraction_complete(f"P{number:02d}")
        for number in range(1, 56)
    ))


def default_worker_counts():
    cpu_count = os.cpu_count() or 4

    extraction_workers = min(
        6,
        max(1, cpu_count // 3),
    )
    experiment_workers = min(
        3,
        max(1, cpu_count // 4),
    )

    return extraction_workers, experiment_workers


def main():
    ensure_directories()

    (
        default_extraction_workers,
        default_experiment_workers,
    ) = default_worker_counts()

    parser = argparse.ArgumentParser(
        description=(
            "Memory-aware CPU pipeline for PhysMelt."
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "preflight",
            "extract",
            "cache",
            "experiments",
            "aggregate",
            "all",
        ],
        default="preflight",
    )

    parser.add_argument(
        "--extract-workers",
        type=int,
        default=default_extraction_workers,
    )
    parser.add_argument(
        "--experiment-workers",
        type=int,
        default=default_experiment_workers,
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=28.0,
        help=(
            "RAM kept available for other workloads."
        ),
    )
    parser.add_argument(
        "--emergency-gb",
        type=float,
        default=16.0,
        help=(
            "Pause PhysMelt workers below this RAM."
        ),
    )
    parser.add_argument(
        "--resume-gb",
        type=float,
        default=24.0,
        help=(
            "Resume paused workers above this RAM."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a reduced smoke experiment."
        ),
    )
    parser.add_argument(
        "--internal-experiment",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    arguments = parser.parse_args()

    if arguments.internal_experiment:
        try:
            run_single_experiment(
                arguments.internal_experiment
            )
        except Exception:
            traceback.print_exc()
        return

    preflight()

    if arguments.stage == "preflight":
        return

    if arguments.stage in {
        "extract",
        "all",
    }:
        extraction_ok = run_extraction(arguments)

        if not extraction_ok:
            print(
                "Extraction finished with failures. "
                "Re-running the same command will resume."
            )

            if arguments.stage == "all":
                return

    if arguments.stage in {
        "cache",
        "all",
    }:
        cache_ok = build_cache()

        if not cache_ok:
            print(
                "Cache construction was not completed."
            )
            return

    if arguments.stage in {
        "experiments",
        "all",
    }:
        experiments_ok = run_experiments(arguments)

        if not experiments_ok:
            print(
                "Some experiment jobs were not completed. "
                "Re-running resumes unfinished jobs."
            )

            if arguments.stage == "all":
                return

    if arguments.stage in {
        "aggregate",
        "all",
    }:
        aggregate_results()

    print()
    print("PhysMelt pipeline command finished.")
    print("The terminal remains open.")


if __name__ == "__main__":
    main()
