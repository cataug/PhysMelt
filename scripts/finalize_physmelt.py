#!/usr/bin/env python3

import os

# Не использовать GPU и не плодить скрытые BLAS-потоки.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture


ROOT = Path.home() / "PhysMelt"

PIPELINE_SCRIPT = ROOT / "scripts/run_physmelt_full.py"
EXTRACTOR_SCRIPT = ROOT / "scripts/extract_frame_features.py"

CACHE_DIR = ROOT / "data/model_cache"
VIDEO_DIR = ROOT / "data/raw/NIST_RHF/MPM_AVIs"

PHYSICS_DIR = ROOT / "outputs/physical_validation"
OUT_DIR = ROOT / "outputs/final_evaluation"
FIGURE_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
FRAME_DIR = OUT_DIR / "qualitative_frames"

OOF_GMM = PHYSICS_DIR / "oof_basic_gmm_seed42.npz"
OFFICIAL_DATA = PHYSICS_DIR / "official_synchronized_data.npy"

PHYSICAL_FPS = 20000.0
FRAME_TIME_MS = 1000.0 / PHYSICAL_FPS

ALARM_THRESHOLDS = [0.30, 0.50, 0.70, 0.90]
EVENT_MIN_LENGTH = 2

BASIC_FEATURES = [
    "threshold_area",
    "mean_intensity",
]


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )

    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return

    fields = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(pipeline):
    required = [
        OOF_GMM,
        OFFICIAL_DATA,
        CACHE_DIR / "metadata.json",
        CACHE_DIR / "features.npy",
        CACHE_DIR / "active.npy",
        CACHE_DIR / "video_index.npy",
        CACHE_DIR / "frame_index.npy",
        CACHE_DIR / "split_code.npy",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("Missing required files:")

        for path in missing:
            print(" ", path)

        raise RuntimeError("Required PhysMelt outputs are missing")

    cache = pipeline.load_cache()

    official = np.load(
        OFFICIAL_DATA,
        mmap_mode="r",
    )

    predictions = np.load(OOF_GMM)

    raw_label = np.asarray(
        predictions["raw_label"],
        dtype=np.int8,
    )
    temporal_label = np.asarray(
        predictions["temporal_label"],
        dtype=np.int8,
    )
    probabilities = np.asarray(
        predictions["probabilities"],
        dtype=np.float64,
    )

    return (
        cache,
        official,
        raw_label,
        temporal_label,
        probabilities,
    )


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, score))


def binary_frame_metrics(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / max(tn + fp, 1)
        ),
        "false_positive_rate": float(
            fp / max(fp + tn, 1)
        ),
        "positive_fraction": float(
            np.mean(y_pred)
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def contiguous_events(binary, frames, minimum_length=2):
    binary = np.asarray(binary, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)

    events = []
    start = None
    previous_frame = None

    for flag, frame in zip(binary, frames):
        contiguous = (
            previous_frame is not None
            and frame == previous_frame + 1
        )

        if flag:
            if start is None or not contiguous:
                if start is not None:
                    end = previous_frame
                    length = end - start + 1

                    if length >= minimum_length:
                        events.append((int(start), int(end)))

                start = frame

        else:
            if start is not None:
                end = previous_frame
                length = end - start + 1

                if length >= minimum_length:
                    events.append((int(start), int(end)))

                start = None

        previous_frame = frame

    if start is not None:
        end = previous_frame
        length = end - start + 1

        if length >= minimum_length:
            events.append((int(start), int(end)))

    return events


def events_overlap(first, second):
    return (
        first[0] <= second[1]
        and second[0] <= first[1]
    )


def event_metrics(
    y_true,
    y_pred,
    video_index,
    frame_index,
):
    true_event_count = 0
    predicted_event_count = 0
    detected_true_events = 0
    false_alarm_events = 0
    detection_delays = []

    videos = np.unique(video_index)

    for video in videos:
        positions = np.flatnonzero(
            video_index == video
        )

        positions = positions[
            np.argsort(frame_index[positions])
        ]

        frames = frame_index[positions]

        true_events = contiguous_events(
            y_true[positions],
            frames,
            minimum_length=EVENT_MIN_LENGTH,
        )

        predicted_events = contiguous_events(
            y_pred[positions],
            frames,
            minimum_length=EVENT_MIN_LENGTH,
        )

        true_event_count += len(true_events)
        predicted_event_count += len(predicted_events)

        for true_event in true_events:
            overlapping = [
                predicted_event
                for predicted_event in predicted_events
                if events_overlap(
                    true_event,
                    predicted_event,
                )
            ]

            if overlapping:
                detected_true_events += 1

                first_prediction_start = min(
                    event[0]
                    for event in overlapping
                )

                delay_frames = max(
                    0,
                    first_prediction_start
                    - true_event[0],
                )

                detection_delays.append(delay_frames)

        for predicted_event in predicted_events:
            has_match = any(
                events_overlap(
                    predicted_event,
                    true_event,
                )
                for true_event in true_events
            )

            if not has_match:
                false_alarm_events += 1

    return {
        "true_events": int(true_event_count),
        "predicted_events": int(predicted_event_count),
        "detected_true_events": int(detected_true_events),
        "event_recall": float(
            detected_true_events
            / max(true_event_count, 1)
        ),
        "false_alarm_events": int(false_alarm_events),
        "false_alarm_events_per_part": float(
            false_alarm_events
            / max(len(videos), 1)
        ),
        "mean_detection_delay_frames": float(
            np.mean(detection_delays)
            if detection_delays
            else np.nan
        ),
        "mean_detection_delay_ms": float(
            FRAME_TIME_MS
            * np.mean(detection_delays)
            if detection_delays
            else np.nan
        ),
    }


def best_f1_threshold(y_true, score):
    thresholds = np.linspace(0.05, 0.95, 91)

    best = {
        "threshold": 0.5,
        "f1": -1.0,
    }

    for threshold in thresholds:
        prediction = score >= threshold

        value = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if value > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(value),
            }

    return best


def evaluate_alarm_definition(
    definition_name,
    y_true,
    score,
    raw_label,
    temporal_label,
    video_index,
    frame_index,
):
    rows = []

    auc = safe_auc(y_true, score)
    auprc = safe_auprc(y_true, score)
    best = best_f1_threshold(y_true, score)

    methods = [
        (
            "gmm_probability_0.30",
            score >= 0.30,
            0.30,
        ),
        (
            "gmm_probability_0.50",
            score >= 0.50,
            0.50,
        ),
        (
            "gmm_probability_0.70",
            score >= 0.70,
            0.70,
        ),
        (
            "gmm_probability_0.90",
            score >= 0.90,
            0.90,
        ),
        (
            "gmm_probability_best_f1_descriptive",
            score >= best["threshold"],
            best["threshold"],
        ),
        (
            "raw_gmm_high_state",
            raw_label == 2,
            np.nan,
        ),
        (
            "temporal_gmm_high_state",
            temporal_label == 2,
            np.nan,
        ),
    ]

    for method_name, prediction, threshold in methods:
        frame_result = binary_frame_metrics(
            y_true,
            prediction,
        )

        event_result = event_metrics(
            y_true,
            prediction,
            video_index,
            frame_index,
        )

        row = {
            "alarm_definition": definition_name,
            "method": method_name,
            "score_threshold": threshold,
            "positive_power_frame_fraction": float(
                np.mean(y_true)
            ),
            "auroc_probability_score": auc,
            "auprc_probability_score": auprc,
            "best_f1_threshold_descriptive": (
                best["threshold"]
            ),
            **frame_result,
            **event_result,
        }

        rows.append(row)

    return rows


def run_alarm_evaluation(
    cache,
    official,
    raw_label,
    temporal_label,
    probabilities,
):
    print()
    print("=" * 78)
    print("ALARM EVALUATION")
    print("=" * 78)

    active = np.asarray(cache["active"])
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    power = np.asarray(
        official[:, 2],
        dtype=np.float64,
    )

    score = probabilities[:, 2]

    valid = (
        active
        & (raw_label >= 0)
        & (temporal_label >= 0)
        & np.isfinite(power)
        & np.isfinite(score)
    )

    power_valid = power[valid]

    top_10_threshold = float(
        np.quantile(power_valid, 0.90)
    )

    definitions = {
        "top_10_percent_measured_power": (
            power_valid >= top_10_threshold
        ),
        "measured_power_at_least_180W": (
            power_valid >= 180.0
        ),
    }

    rows = []

    for name, y_true in definitions.items():
        print()
        print("Definition:", name)
        print("Positive frames:", int(y_true.sum()))
        print(
            "Positive fraction:",
            f"{100.0 * y_true.mean():.2f}%",
        )

        current_rows = evaluate_alarm_definition(
            name,
            y_true,
            score[valid],
            raw_label[valid],
            temporal_label[valid],
            video_index[valid],
            frame_index[valid],
        )

        rows.extend(current_rows)

        probability_row = next(
            row
            for row in current_rows
            if row["method"] == "gmm_probability_0.50"
        )

        temporal_row = next(
            row
            for row in current_rows
            if row["method"]
            == "temporal_gmm_high_state"
        )

        print(
            "AUROC:",
            f"{probability_row['auroc_probability_score']:.3f}",
        )
        print(
            "AUPRC:",
            f"{probability_row['auprc_probability_score']:.3f}",
        )
        print(
            "P(high)>=0.5 F1:",
            f"{probability_row['f1']:.3f}",
        )
        print(
            "Temporal-state F1:",
            f"{temporal_row['f1']:.3f}",
        )
        print(
            "Temporal event recall:",
            f"{temporal_row['event_recall']:.3f}",
        )
        print(
            "Temporal false alarms per part:",
            f"{temporal_row['false_alarm_events_per_part']:.3f}",
        )

    write_csv(
        OUT_DIR / "alarm_evaluation.csv",
        rows,
    )

    summary = {
        "top_10_percent_power_threshold_w": (
            top_10_threshold
        ),
        "event_minimum_length_frames": (
            EVENT_MIN_LENGTH
        ),
        "frame_time_ms": FRAME_TIME_MS,
        "rows": rows,
    }

    write_json(
        OUT_DIR / "alarm_evaluation.json",
        summary,
    )

    return summary


def fit_scaler(train_features):
    medians = np.nanmedian(
        np.where(
            np.isfinite(train_features),
            train_features,
            np.nan,
        ),
        axis=0,
    )

    medians = np.where(
        np.isfinite(medians),
        medians,
        0.0,
    )

    filled = np.asarray(
        train_features,
        dtype=np.float64,
    ).copy()

    invalid = ~np.isfinite(filled)

    if invalid.any():
        row_ids, column_ids = np.where(invalid)
        filled[row_ids, column_ids] = medians[column_ids]

    means = filled.mean(axis=0)
    standard_deviations = filled.std(axis=0)

    standard_deviations = np.where(
        standard_deviations < 1e-9,
        1.0,
        standard_deviations,
    )

    return medians, means, standard_deviations


def apply_scaler(
    features,
    medians,
    means,
    standard_deviations,
):
    result = np.asarray(
        features,
        dtype=np.float64,
    ).copy()

    invalid = ~np.isfinite(result)

    if invalid.any():
        row_ids, column_ids = np.where(invalid)
        result[row_ids, column_ids] = medians[column_ids]

    return (
        result - means
    ) / standard_deviations


def benchmark_batch(function, item_count, repetitions=20):
    function()

    times = []

    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    median_seconds = float(np.median(times))
    p95_seconds = float(np.quantile(times, 0.95))

    return {
        "items": int(item_count),
        "repetitions": int(repetitions),
        "median_total_ms": (
            1000.0 * median_seconds
        ),
        "p95_total_ms": (
            1000.0 * p95_seconds
        ),
        "median_us_per_frame": (
            1e6
            * median_seconds
            / max(item_count, 1)
        ),
        "p95_us_per_frame": (
            1e6
            * p95_seconds
            / max(item_count, 1)
        ),
        "median_fps": float(
            item_count
            / max(median_seconds, 1e-12)
        ),
    }


def reorder_gmm(gmm):
    thermal_order = np.argsort(
        gmm.means_.sum(axis=1)
    )

    return thermal_order


def benchmark_raw_minimal_features(
    extractor,
    cache,
):
    metadata = cache["metadata"]
    video_names = metadata["video_names"]
    feature_names = metadata["feature_names"]

    area_column = feature_names.index(
        "threshold_area"
    )
    intensity_column = feature_names.index(
        "mean_intensity"
    )

    cached_features = cache["features"]
    cached_video_index = np.asarray(
        cache["video_index"]
    )
    cached_frame_index = np.asarray(
        cache["frame_index"]
    )

    selected_video_names = video_names[:3]
    maximum_frames_per_video = 500

    decoded_count = 0
    absolute_area_errors = []
    absolute_intensity_errors = []

    start = time.perf_counter()

    for current_video_name in selected_video_names:
        match = re.search(
            r"P(\d+)$",
            current_video_name,
        )

        part_number = int(match.group(1))
        video_path = (
            VIDEO_DIR
            / f"RHF_MPM_P{part_number:02d}.avi"
        )

        current_video_index = video_names.index(
            current_video_name
        )

        with video_path.open("rb") as handle:
            movi_start, movi_end = extractor.find_movi(
                handle,
                video_path.stat().st_size,
            )

            frame_chunks = []

            extractor.collect_frame_chunks(
                handle,
                movi_start,
                movi_end,
                frame_chunks,
            )

            limit = min(
                maximum_frames_per_video,
                len(frame_chunks),
            )

            for frame_id in range(limit):
                offset, size = frame_chunks[frame_id]

                image = extractor.decode_gray_frame(
                    handle,
                    offset,
                    size,
                )

                histogram = np.bincount(
                    image.ravel(),
                    minlength=256,
                )

                otsu = extractor.otsu_threshold(
                    histogram
                )

                threshold = max(8, int(otsu))

                area = int(
                    np.count_nonzero(
                        image >= threshold
                    )
                )

                mean_intensity = float(
                    image.mean()
                )

                position = np.flatnonzero(
                    (
                        cached_video_index
                        == current_video_index
                    )
                    & (
                        cached_frame_index
                        == frame_id
                    )
                )

                if len(position) == 1:
                    cached_row = position[0]

                    absolute_area_errors.append(
                        abs(
                            area
                            - float(
                                cached_features[
                                    cached_row,
                                    area_column,
                                ]
                            )
                        )
                    )

                    absolute_intensity_errors.append(
                        abs(
                            mean_intensity
                            - float(
                                cached_features[
                                    cached_row,
                                    intensity_column,
                                ]
                            )
                        )
                    )

                decoded_count += 1

    elapsed = time.perf_counter() - start

    return {
        "videos": selected_video_names,
        "frames": int(decoded_count),
        "total_seconds": float(elapsed),
        "ms_per_frame": float(
            1000.0
            * elapsed
            / max(decoded_count, 1)
        ),
        "fps": float(
            decoded_count
            / max(elapsed, 1e-12)
        ),
        "area_mae_against_cache_pixels": float(
            np.mean(absolute_area_errors)
            if absolute_area_errors
            else np.nan
        ),
        "intensity_mae_against_cache_dl": float(
            np.mean(absolute_intensity_errors)
            if absolute_intensity_errors
            else np.nan
        ),
    }


def run_edge_benchmark(
    extractor,
    cache,
    temporal_label,
):
    print()
    print("=" * 78)
    print("EDGE / DEPLOYMENT BENCHMARK")
    print("=" * 78)

    metadata = cache["metadata"]
    feature_names = metadata["feature_names"]

    columns = [
        feature_names.index(name)
        for name in BASIC_FEATURES
    ]

    matrix = np.asarray(
        cache["features"][:, columns],
        dtype=np.float64,
    )

    active = np.asarray(cache["active"])
    split_code = np.asarray(cache["split_code"])

    train_mask = (
        active
        & (split_code != 2)
        & (temporal_label >= 0)
    )

    test_mask = (
        active
        & (split_code == 2)
        & (temporal_label >= 0)
    )

    train_raw = matrix[train_mask]
    test_raw = matrix[test_mask]

    y_train = temporal_label[train_mask]
    y_test = temporal_label[test_mask]

    (
        medians,
        means,
        standard_deviations,
    ) = fit_scaler(train_raw)

    train_features = apply_scaler(
        train_raw,
        medians,
        means,
        standard_deviations,
    )

    test_features = apply_scaler(
        test_raw,
        medians,
        means,
        standard_deviations,
    )

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="diag",
        reg_covar=1e-5,
        max_iter=500,
        n_init=5,
        random_state=42,
    )

    gmm.fit(train_features)

    component_order = reorder_gmm(gmm)

    gmm_test_probabilities = (
        gmm.predict_proba(test_features)[
            :,
            component_order,
        ]
    )

    gmm_test_labels = np.argmax(
        gmm_test_probabilities,
        axis=1,
    )

    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    logistic.fit(
        train_features,
        y_train,
    )

    logistic_prediction = logistic.predict(
        test_features
    )

    logistic_probabilities = (
        logistic.predict_proba(test_features)
    )

    gmm_benchmark = benchmark_batch(
        lambda: (
            gmm.predict_proba(test_features)[
                :,
                component_order,
            ]
        ),
        len(test_features),
        repetitions=30,
    )

    logistic_benchmark = benchmark_batch(
        lambda: logistic.predict_proba(
            test_features
        ),
        len(test_features),
        repetitions=30,
    )

    end_to_end_gmm_benchmark = benchmark_batch(
        lambda: (
            gmm.predict_proba(
                apply_scaler(
                    test_raw,
                    medians,
                    means,
                    standard_deviations,
                )
            )[:, component_order]
        ),
        len(test_features),
        repetitions=20,
    )

    raw_benchmark = benchmark_raw_minimal_features(
        extractor,
        cache,
    )

    gmm_model_path = MODEL_DIR / "physmelt_lite_gmm.npz"

    np.savez_compressed(
        gmm_model_path,
        feature_names=np.asarray(BASIC_FEATURES),
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        weights=gmm.weights_[component_order],
        component_means=gmm.means_[component_order],
        component_covariances=(
            gmm.covariances_[component_order]
        ),
        physical_fps=np.asarray([PHYSICAL_FPS]),
    )

    logistic_model_path = (
        MODEL_DIR
        / "physmelt_lite_logistic_distilled.npz"
    )

    np.savez_compressed(
        logistic_model_path,
        feature_names=np.asarray(BASIC_FEATURES),
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        coefficients=logistic.coef_,
        intercept=logistic.intercept_,
        classes=logistic.classes_,
    )

    result = {
        "train_frames": int(len(train_features)),
        "test_frames": int(len(test_features)),
        "feature_count": len(BASIC_FEATURES),
        "feature_names": BASIC_FEATURES,
        "gmm_parameter_count": int(
            gmm.weights_.size
            + gmm.means_.size
            + gmm.covariances_.size
            + medians.size
            + means.size
            + standard_deviations.size
        ),
        "gmm_model_file": str(gmm_model_path),
        "gmm_model_size_bytes": (
            gmm_model_path.stat().st_size
        ),
        "gmm_agreement_with_temporal_oof": float(
            accuracy_score(
                y_test,
                gmm_test_labels,
            )
        ),
        "gmm_balanced_agreement": float(
            balanced_accuracy_score(
                y_test,
                gmm_test_labels,
            )
        ),
        "gmm_macro_f1_agreement": float(
            f1_score(
                y_test,
                gmm_test_labels,
                average="macro",
            )
        ),
        "gmm_probability_checksum": float(
            gmm_test_probabilities.mean()
        ),
        "gmm_inference": gmm_benchmark,
        "gmm_scaling_plus_inference": (
            end_to_end_gmm_benchmark
        ),
        "logistic_parameter_count": int(
            logistic.coef_.size
            + logistic.intercept_.size
            + medians.size
            + means.size
            + standard_deviations.size
        ),
        "logistic_model_file": str(
            logistic_model_path
        ),
        "logistic_model_size_bytes": (
            logistic_model_path.stat().st_size
        ),
        "logistic_accuracy": float(
            accuracy_score(
                y_test,
                logistic_prediction,
            )
        ),
        "logistic_balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                logistic_prediction,
            )
        ),
        "logistic_macro_f1": float(
            f1_score(
                y_test,
                logistic_prediction,
                average="macro",
            )
        ),
        "logistic_probability_checksum": float(
            logistic_probabilities.mean()
        ),
        "logistic_inference": logistic_benchmark,
        "raw_decode_and_minimal_features": (
            raw_benchmark
        ),
    }

    write_json(
        OUT_DIR / "edge_benchmark.json",
        result,
    )

    print("Basic features:", BASIC_FEATURES)
    print(
        "GMM model size:",
        result["gmm_model_size_bytes"],
        "bytes",
    )
    print(
        "GMM parameters:",
        result["gmm_parameter_count"],
    )
    print(
        "GMM inference:",
        f"{gmm_benchmark['median_us_per_frame']:.3f}",
        "us/frame",
    )
    print(
        "GMM inference FPS:",
        f"{gmm_benchmark['median_fps']:.0f}",
    )
    print(
        "Raw decode + minimal features:",
        f"{raw_benchmark['ms_per_frame']:.3f}",
        "ms/frame",
    )
    print(
        "Raw decode + minimal feature FPS:",
        f"{raw_benchmark['fps']:.1f}",
    )
    print(
        "Logistic agreement macro-F1:",
        f"{result['logistic_macro_f1']:.3f}",
    )

    return result


def rolling_statistic(values, window, mode):
    values = np.asarray(values, dtype=np.float64)

    if len(values) <= window:
        return np.asarray([
            np.mean(values)
            if mode == "mean"
            else np.sum(values)
        ])

    kernel = np.ones(window, dtype=np.float64)

    if mode == "mean":
        kernel /= window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def choose_window(
    mode,
    power,
    p_high,
    raw_label,
    temporal_label,
    frame_index,
    window_length=300,
):
    count = len(power)

    if count <= window_length:
        return 0, count

    if mode == "high_power":
        statistic = rolling_statistic(
            power,
            window_length,
            "mean",
        )

    elif mode == "temporal_correction":
        disagreement = (
            raw_label != temporal_label
        ).astype(np.float64)

        statistic = rolling_statistic(
            disagreement,
            window_length,
            "sum",
        )

    elif mode == "transition":
        transitions = np.zeros(
            count,
            dtype=np.float64,
        )

        transitions[1:] = (
            temporal_label[1:]
            != temporal_label[:-1]
        )

        high_entries = np.zeros(
            count,
            dtype=np.float64,
        )

        high_entries[1:] = (
            (temporal_label[1:] == 2)
            & (temporal_label[:-1] != 2)
        )

        statistic = rolling_statistic(
            transitions + 3.0 * high_entries,
            window_length,
            "sum",
        )

    else:
        centered = (
            p_high
            - np.nanmean(p_high)
        ) ** 2

        statistic = rolling_statistic(
            centered,
            window_length,
            "mean",
        )

    start = int(np.nanargmax(statistic))
    end = min(start + window_length, count)

    return start, end


def select_qualitative_cases(
    cache,
    official,
    raw_label,
    temporal_label,
    probabilities,
):
    active = np.asarray(cache["active"])
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    power = np.asarray(official[:, 2])
    p_high = probabilities[:, 2]

    records = []

    for video in np.unique(video_index):
        positions = np.flatnonzero(
            (
                video_index == video
            )
            & active
            & (temporal_label >= 0)
            & np.isfinite(power)
        )

        positions = positions[
            np.argsort(frame_index[positions])
        ]

        if len(positions) < 50:
            continue

        video_power = power[positions]
        video_score = p_high[positions]

        disagreement_fraction = float(
            np.mean(
                raw_label[positions]
                != temporal_label[positions]
            )
        )

        transitions = int(
            np.sum(
                temporal_label[positions][1:]
                != temporal_label[positions][:-1]
            )
        )

        if (
            np.std(video_power) > 1e-9
            and np.std(video_score) > 1e-9
        ):
            correlation = float(
                np.corrcoef(
                    video_power,
                    video_score,
                )[0, 1]
            )
        else:
            correlation = np.nan

        rolling_power = rolling_statistic(
            video_power,
            min(150, len(video_power)),
            "mean",
        )

        records.append({
            "video_index": int(video),
            "positions": positions,
            "maximum_rolling_power": float(
                np.max(rolling_power)
            ),
            "disagreement_fraction": (
                disagreement_fraction
            ),
            "transitions": transitions,
            "score_power_correlation": correlation,
        })

    selected = []

    highest_power = max(
        records,
        key=lambda item: item[
            "maximum_rolling_power"
        ],
    )
    selected.append(
        ("high_power", highest_power)
    )

    correction_candidates = [
        item
        for item in records
        if item["video_index"]
        != highest_power["video_index"]
    ]

    highest_correction = max(
        correction_candidates,
        key=lambda item: item[
            "disagreement_fraction"
        ],
    )
    selected.append(
        (
            "temporal_correction",
            highest_correction,
        )
    )

    used = {
        item["video_index"]
        for _, item in selected
    }

    transition_candidates = [
        item
        for item in records
        if item["video_index"] not in used
    ]

    transition_case = max(
        transition_candidates,
        key=lambda item: item["transitions"],
    )
    selected.append(
        ("transition", transition_case)
    )

    used.add(transition_case["video_index"])

    remaining = [
        item
        for item in records
        if item["video_index"] not in used
        and np.isfinite(
            item["score_power_correlation"]
        )
    ]

    remaining.sort(
        key=lambda item: item[
            "score_power_correlation"
        ]
    )

    representative = remaining[
        len(remaining) // 2
    ]

    selected.append(
        ("representative", representative)
    )

    return selected


def decode_selected_frames(
    extractor,
    video_path,
    requested_frames,
):
    images = {}

    with video_path.open("rb") as handle:
        movi_start, movi_end = extractor.find_movi(
            handle,
            video_path.stat().st_size,
        )

        chunks = []

        extractor.collect_frame_chunks(
            handle,
            movi_start,
            movi_end,
            chunks,
        )

        for frame_id in requested_frames:
            if frame_id < 0 or frame_id >= len(chunks):
                continue

            offset, size = chunks[frame_id]

            images[frame_id] = (
                extractor.decode_gray_frame(
                    handle,
                    offset,
                    size,
                )
            )

    return images


def choose_snapshot_frames(
    positions,
    frame_index,
    temporal_label,
    probabilities,
):
    selected = []

    for regime in range(3):
        regime_positions = positions[
            temporal_label[positions] == regime
        ]

        if len(regime_positions):
            confidence = probabilities[
                regime_positions,
                regime,
            ]

            selected_position = regime_positions[
                int(np.nanargmax(confidence))
            ]
        else:
            selected_position = positions[
                len(positions) // 2
            ]

        selected.append(
            int(frame_index[selected_position])
        )

    return selected


def make_qualitative_figures(
    extractor,
    cache,
    official,
    raw_label,
    temporal_label,
    probabilities,
):
    print()
    print("=" * 78)
    print("QUALITATIVE TEMPORAL PANELS")
    print("=" * 78)

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    metadata = cache["metadata"]
    video_names = metadata["video_names"]
    feature_names = metadata["feature_names"]

    area_column = feature_names.index(
        "threshold_area"
    )
    intensity_column = feature_names.index(
        "mean_intensity"
    )

    features = cache["features"]
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    power = np.asarray(official[:, 2])
    p_high = probabilities[:, 2]

    cases = select_qualitative_cases(
        cache,
        official,
        raw_label,
        temporal_label,
        probabilities,
    )

    manifest_rows = []

    for case_number, (
        case_name,
        case,
    ) in enumerate(cases, start=1):

        current_video_index = case["video_index"]
        video_name = video_names[
            current_video_index
        ]

        positions = case["positions"]

        mode = (
            case_name
            if case_name in {
                "high_power",
                "temporal_correction",
                "transition",
            }
            else "representative"
        )

        start_local, end_local = choose_window(
            mode,
            power[positions],
            p_high[positions],
            raw_label[positions],
            temporal_label[positions],
            frame_index[positions],
            window_length=300,
        )

        window_positions = positions[
            start_local:end_local
        ]

        snapshot_frames = choose_snapshot_frames(
            window_positions,
            frame_index,
            temporal_label,
            probabilities,
        )

        match = re.search(
            r"P(\d+)$",
            video_name,
        )
        part_number = int(match.group(1))

        video_path = (
            VIDEO_DIR
            / f"RHF_MPM_P{part_number:02d}.avi"
        )

        images = decode_selected_frames(
            extractor,
            video_path,
            snapshot_frames,
        )

        time_ms = (
            frame_index[window_positions]
            / PHYSICAL_FPS
            * 1000.0
        )

        figure = plt.figure(
            figsize=(12, 12)
        )

        grid = GridSpec(
            6,
            3,
            figure=figure,
            height_ratios=[
                1.25,
                1.0,
                1.0,
                1.0,
                0.8,
                0.8,
            ],
        )

        regime_names = [
            "Low thermal",
            "Nominal",
            "High thermal",
        ]

        for regime, frame_id in enumerate(
            snapshot_frames
        ):
            axis = figure.add_subplot(
                grid[0, regime]
            )

            image = images.get(frame_id)

            if image is not None:
                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )

            axis.set_title(
                f"{regime_names[regime]}\n"
                f"frame {frame_id}"
            )
            axis.set_axis_off()

            if image is not None:
                frame_path = (
                    FRAME_DIR
                    / (
                        f"{case_number:02d}_"
                        f"{case_name}_"
                        f"P{part_number:02d}_"
                        f"state{regime}_"
                        f"frame{frame_id}.npy"
                    )
                )

                np.save(
                    frame_path,
                    image,
                )

        power_axis = figure.add_subplot(
            grid[1, :]
        )

        power_axis.plot(
            time_ms,
            power[window_positions],
        )
        power_axis.axhline(
            180.0,
            linestyle="--",
            label="180 W alarm reference",
        )
        power_axis.set_ylabel(
            "Measured power (W)"
        )
        power_axis.legend(
            loc="best"
        )
        power_axis.grid(alpha=0.25)

        feature_axis = figure.add_subplot(
            grid[2, :]
        )

        feature_axis.plot(
            time_ms,
            features[
                window_positions,
                area_column,
            ],
            label="Area (pixels)",
        )

        second_feature_axis = (
            feature_axis.twinx()
        )

        second_feature_axis.plot(
            time_ms,
            features[
                window_positions,
                intensity_column,
            ],
            label="Mean intensity",
        )

        feature_axis.set_ylabel(
            "Melt-pool area (pixels)"
        )
        second_feature_axis.set_ylabel(
            "Mean intensity (DL)"
        )
        feature_axis.grid(alpha=0.25)

        handles_first, labels_first = (
            feature_axis.get_legend_handles_labels()
        )

        handles_second, labels_second = (
            second_feature_axis
            .get_legend_handles_labels()
        )

        feature_axis.legend(
            handles_first + handles_second,
            labels_first + labels_second,
            loc="best",
        )

        probability_axis = figure.add_subplot(
            grid[3, :]
        )

        probability_axis.plot(
            time_ms,
            p_high[window_positions],
            label=r"$P(\mathrm{high})$",
        )

        probability_axis.axhline(
            0.5,
            linestyle="--",
            label="0.5 threshold",
        )

        probability_axis.set_ylim(
            -0.03,
            1.03,
        )
        probability_axis.set_ylabel(
            "High-regime probability"
        )
        probability_axis.legend(
            loc="best"
        )
        probability_axis.grid(alpha=0.25)

        raw_axis = figure.add_subplot(
            grid[4, :]
        )

        raw_axis.step(
            time_ms,
            raw_label[window_positions],
            where="post",
        )
        raw_axis.set_ylabel(
            "Raw state"
        )
        raw_axis.set_yticks(
            [0, 1, 2],
            labels=["Low", "Nom.", "High"],
        )
        raw_axis.grid(alpha=0.25)

        temporal_axis = figure.add_subplot(
            grid[5, :]
        )

        temporal_axis.step(
            time_ms,
            temporal_label[
                window_positions
            ],
            where="post",
        )
        temporal_axis.set_ylabel(
            "Temporal state"
        )
        temporal_axis.set_yticks(
            [0, 1, 2],
            labels=["Low", "Nom.", "High"],
        )
        temporal_axis.set_xlabel(
            "Physical time (ms)"
        )
        temporal_axis.grid(alpha=0.25)

        disagreement_fraction = float(
            np.mean(
                raw_label[window_positions]
                != temporal_label[
                    window_positions
                ]
            )
        )

        figure.suptitle(
            f"{case_name.replace('_', ' ').title()}: "
            f"P{part_number:02d} — "
            f"temporal corrections "
            f"{100.0 * disagreement_fraction:.1f}%",
            fontsize=15,
        )

        figure.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        base_name = (
            f"{case_number:02d}_"
            f"{case_name}_"
            f"P{part_number:02d}"
        )

        pdf_path = (
            FIGURE_DIR
            / f"{base_name}.pdf"
        )
        png_path = (
            FIGURE_DIR
            / f"{base_name}.png"
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        figure.savefig(
            png_path,
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(figure)

        manifest_rows.append({
            "case": case_name,
            "video_id": video_name,
            "part_number": part_number,
            "window_start_frame": int(
                frame_index[
                    window_positions[0]
                ]
            ),
            "window_end_frame": int(
                frame_index[
                    window_positions[-1]
                ]
            ),
            "window_duration_ms": float(
                (
                    frame_index[
                        window_positions[-1]
                    ]
                    - frame_index[
                        window_positions[0]
                    ]
                )
                * FRAME_TIME_MS
            ),
            "window_disagreement_fraction": (
                disagreement_fraction
            ),
            "snapshot_low_frame": (
                snapshot_frames[0]
            ),
            "snapshot_nominal_frame": (
                snapshot_frames[1]
            ),
            "snapshot_high_frame": (
                snapshot_frames[2]
            ),
            "pdf": str(pdf_path),
            "png": str(png_path),
        })

        print(
            f"[{case_number}/4] "
            f"{case_name}: "
            f"P{part_number:02d}, "
            f"frames "
            f"{manifest_rows[-1]['window_start_frame']}"
            f"-"
            f"{manifest_rows[-1]['window_end_frame']}"
        )

    write_csv(
        OUT_DIR / "qualitative_manifest.csv",
        manifest_rows,
    )

    return manifest_rows


def write_final_markdown(
    alarm_summary,
    edge_summary,
    qualitative_manifest,
):
    path = (
        OUT_DIR
        / "final_completion_summary.md"
    )

    alarm_rows = alarm_summary["rows"]

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# PhysMelt final evaluation\n\n"
        )

        handle.write(
            "## Alarm evaluation\n\n"
        )

        handle.write(
            "| Definition | Method | AUROC | AUPRC | "
            "Precision | Recall | F1 | Event recall | "
            "False alarms/part |\n"
        )

        handle.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        preferred_methods = {
            "gmm_probability_0.50",
            "gmm_probability_0.70",
            "temporal_gmm_high_state",
        }

        for row in alarm_rows:
            if row["method"] not in preferred_methods:
                continue

            handle.write(
                f"| {row['alarm_definition']} "
                f"| {row['method']} "
                f"| {row['auroc_probability_score']:.3f} "
                f"| {row['auprc_probability_score']:.3f} "
                f"| {row['precision']:.3f} "
                f"| {row['recall']:.3f} "
                f"| {row['f1']:.3f} "
                f"| {row['event_recall']:.3f} "
                f"| {row['false_alarm_events_per_part']:.2f} |\n"
            )

        handle.write(
            "\n## Edge benchmark\n\n"
        )

        handle.write(
            f"- Basic features: "
            f"{', '.join(edge_summary['feature_names'])}\n"
        )
        handle.write(
            f"- GMM parameter count: "
            f"{edge_summary['gmm_parameter_count']}\n"
        )
        handle.write(
            f"- Serialized GMM size: "
            f"{edge_summary['gmm_model_size_bytes']} bytes\n"
        )
        handle.write(
            f"- GMM inference latency: "
            f"{edge_summary['gmm_inference']['median_us_per_frame']:.3f} "
            f"us/frame\n"
        )
        handle.write(
            f"- GMM inference throughput: "
            f"{edge_summary['gmm_inference']['median_fps']:.0f} FPS\n"
        )
        handle.write(
            f"- Raw AVI decode plus minimal feature extraction: "
            f"{edge_summary['raw_decode_and_minimal_features']['ms_per_frame']:.3f} "
            f"ms/frame\n"
        )
        handle.write(
            f"- Raw decode plus minimal feature throughput: "
            f"{edge_summary['raw_decode_and_minimal_features']['fps']:.1f} "
            f"FPS\n"
        )
        handle.write(
            f"- Distilled logistic macro-F1 agreement: "
            f"{edge_summary['logistic_macro_f1']:.3f}\n"
        )

        handle.write(
            "\n## Qualitative panels\n\n"
        )

        for row in qualitative_manifest:
            handle.write(
                f"- `{row['case']}`: "
                f"{row['video_id']}, frames "
                f"{row['window_start_frame']}--"
                f"{row['window_end_frame']}, "
                f"correction fraction "
                f"{100.0 * row['window_disagreement_fraction']:.1f}%\n"
            )

        handle.write(
            "\nThe descriptive best-F1 probability threshold is "
            "reported for inspection only; paper claims should rely "
            "primarily on threshold-free AUROC/AUPRC and fixed "
            "operating thresholds.\n"
        )

    return path


def main():
    for directory in [
        OUT_DIR,
        FIGURE_DIR,
        MODEL_DIR,
        FRAME_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Loading PhysMelt modules...")

    pipeline = load_module(
        "physmelt_pipeline",
        PIPELINE_SCRIPT,
    )

    extractor = load_module(
        "physmelt_extractor",
        EXTRACTOR_SCRIPT,
    )

    (
        cache,
        official,
        raw_label,
        temporal_label,
        probabilities,
    ) = load_inputs(pipeline)

    print("Frames:", len(raw_label))
    print(
        "Active frames:",
        int(np.asarray(cache["active"]).sum()),
    )

    alarm_summary = run_alarm_evaluation(
        cache,
        official,
        raw_label,
        temporal_label,
        probabilities,
    )

    edge_summary = run_edge_benchmark(
        extractor,
        cache,
        temporal_label,
    )

    qualitative_manifest = (
        make_qualitative_figures(
            extractor,
            cache,
            official,
            raw_label,
            temporal_label,
            probabilities,
        )
    )

    markdown_path = write_final_markdown(
        alarm_summary,
        edge_summary,
        qualitative_manifest,
    )

    print()
    print("=" * 78)
    print("FINAL PHYS-MELT COMPLETION")
    print("=" * 78)
    print("Alarm results:")
    print(OUT_DIR / "alarm_evaluation.csv")
    print("Edge benchmark:")
    print(OUT_DIR / "edge_benchmark.json")
    print("Qualitative manifest:")
    print(OUT_DIR / "qualitative_manifest.csv")
    print("Paper-ready summary:")
    print(markdown_path)
    print("Figures:")
    print(FIGURE_DIR)
    print()
    print(
        "Final evaluation finished. "
        "The terminal remains open."
    )


if __name__ == "__main__":
    main()
