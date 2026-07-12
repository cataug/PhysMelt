#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import importlib.util
import io
import json
import math
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


ROOT = Path.home() / "PhysMelt"
PIPELINE_PATH = ROOT / "scripts/run_physmelt_full.py"
RAW_DIR = ROOT / "data/raw/NIST_RHF"
CACHE_DIR = ROOT / "data/model_cache"
OUT_DIR = ROOT / "outputs/physical_validation"

SEEDS = [42, 43, 44]
MODELS = ["kmeans", "gmm"]

OFFICIAL_COLUMNS = [
    "x_mm",
    "y_mm",
    "power_w",
    "speed_mm_s",
    "area_mm2",
    "length_mm",
    "width_mm",
    "mean_intensity",
    "spatter_count",
    "spatter_area",
]

PIXEL_SIZE_MM = 0.008
PIXEL_AREA_MM2 = PIXEL_SIZE_MM ** 2


def load_pipeline_module():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline was not found: {PIPELINE_PATH}"
        )

    specification = importlib.util.spec_from_file_location(
        "physmelt_pipeline",
        PIPELINE_PATH,
    )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


def find_analysis_archive():
    matches = [
        path
        for path in RAW_DIR.rglob("*.zip")
        if path.name.lower() == "rhf_analysis_results.zip"
    ]

    if not matches:
        raise FileNotFoundError(
            "RHF_Analysis_Results.zip was not found under "
            f"{RAW_DIR}"
        )

    return matches[0]


def read_numeric_csv(zip_handle, member):
    rows = []

    with zip_handle.open(member) as raw_handle:
        text_handle = io.TextIOWrapper(
            raw_handle,
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        )

        for row in csv.reader(text_handle):
            if not row:
                continue

            try:
                values = [
                    float(value.strip())
                    for value in row[:10]
                ]
            except ValueError:
                continue

            if len(values) == 10:
                rows.append(values)

    return np.asarray(rows, dtype=np.float64)


def load_official_tables(cache):
    archive_path = find_analysis_archive()

    metadata = cache["metadata"]
    video_names = metadata["video_names"]

    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    total_rows = len(video_index)

    official = np.full(
        (total_rows, 10),
        np.nan,
        dtype=np.float64,
    )

    print("Analysis archive:", archive_path)

    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        member_lookup = {}

        for member in zip_handle.namelist():
            match = re.search(
                r"DAQ_RHF_P(\d+)_layer0001_T80_XYPVALWI\.csv$",
                member,
                flags=re.IGNORECASE,
            )

            if match:
                part_number = int(match.group(1))
                member_lookup[part_number] = member

        print("Official synchronized CSV files:", len(member_lookup))

        if len(member_lookup) != 55:
            raise RuntimeError(
                f"Expected 55 official CSV files, found "
                f"{len(member_lookup)}"
            )

        for current_video_index, video_name in enumerate(video_names):
            match = re.search(r"P(\d+)$", video_name)

            if not match:
                raise RuntimeError(
                    f"Could not parse part number: {video_name}"
                )

            part_number = int(match.group(1))
            member = member_lookup[part_number]
            rows = read_numeric_csv(zip_handle, member)

            positions = np.flatnonzero(
                video_index == current_video_index
            )
            positions = positions[
                np.argsort(frame_index[positions])
            ]

            if len(rows) != len(positions):
                raise RuntimeError(
                    f"P{part_number:02d}: official rows={len(rows)}, "
                    f"video frames={len(positions)}"
                )

            if len(rows) != 1498:
                raise RuntimeError(
                    f"P{part_number:02d}: expected 1498 rows, "
                    f"found {len(rows)}"
                )

            official[positions, :] = rows

            print(
                f"[{current_video_index + 1:02d}/55] "
                f"P{part_number:02d}: aligned {len(rows)} rows",
                flush=True,
            )

    np.save(
        OUT_DIR / "official_synchronized_data.npy",
        official,
    )

    return official


def average_rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    position = 0

    while position < len(values):
        end = position + 1
        current_value = values[order[position]]

        while (
            end < len(values)
            and values[order[end]] == current_value
        ):
            end += 1

        average = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average
        position = end

    return ranks


def pearson_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]

    if len(first) < 3:
        return np.nan

    first_std = first.std()
    second_std = second.std()

    if first_std < 1e-12 or second_std < 1e-12:
        return np.nan

    return float(np.corrcoef(first, second)[0, 1])


def spearman_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]

    if len(first) < 3:
        return np.nan

    return pearson_correlation(
        average_rank(first),
        average_rank(second),
    )


def bootstrap_mean_interval(values, seed=42, repetitions=4000):
    values = np.asarray(
        [
            value
            for value in values
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    generator = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)

    for index in range(repetitions):
        sample = generator.choice(
            values,
            size=len(values),
            replace=True,
        )
        bootstrap[index] = sample.mean()

    return (
        float(values.mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )

    temporary.replace(path)


def prepare_features(pm, cache, train_indices, test_indices):
    metadata = cache["metadata"]
    feature_names = metadata["feature_names"]

    selected_names = [
        "threshold_area",
        "mean_intensity",
    ]

    columns = [
        feature_names.index(name)
        for name in selected_names
    ]

    matrix = cache["features"]

    train_features = np.asarray(
        matrix[train_indices][:, columns],
        dtype=np.float64,
    )
    test_features = np.asarray(
        matrix[test_indices][:, columns],
        dtype=np.float64,
    )

    medians = pm.finite_imputer_fit(train_features)

    train_features = pm.finite_imputer_transform(
        train_features,
        medians,
    )
    test_features = pm.finite_imputer_transform(
        test_features,
        medians,
    )

    means, standard_deviations = pm.standardizer_fit(
        train_features
    )

    train_standardized = pm.standardizer_transform(
        train_features,
        means,
        standard_deviations,
    )
    test_standardized = pm.standardizer_transform(
        test_features,
        means,
        standard_deviations,
    )

    train_thermal = (
        train_standardized[:, 0]
        + train_standardized[:, 1]
    )
    test_thermal = (
        test_standardized[:, 0]
        + test_standardized[:, 1]
    )

    return (
        train_standardized,
        test_standardized,
        train_thermal,
        test_thermal,
    )


def fit_fold(
    pm,
    model_name,
    seed,
    train_features,
    test_features,
    train_thermal,
):
    if model_name == "kmeans":
        model = KMeans(
            n_clusters=3,
            n_init=20,
            max_iter=500,
            random_state=seed,
        )

        train_labels_original = model.fit_predict(
            train_features
        )
        test_labels_original = model.predict(test_features)

        train_distances = model.transform(train_features) ** 2
        test_distances = model.transform(test_features) ** 2

        temperature = float(
            np.median(
                np.min(train_distances, axis=1)
            )
        )
        temperature = max(temperature, 1e-3)

        train_probabilities_original = pm.softmax(
            -train_distances / (2.0 * temperature)
        )
        test_probabilities_original = pm.softmax(
            -test_distances / (2.0 * temperature)
        )

    elif model_name == "gmm":
        model = GaussianMixture(
            n_components=3,
            covariance_type="diag",
            reg_covar=1e-5,
            max_iter=500,
            n_init=5,
            random_state=seed,
        )

        model.fit(train_features)

        train_labels_original = model.predict(
            train_features
        )
        test_labels_original = model.predict(
            test_features
        )

        train_probabilities_original = model.predict_proba(
            train_features
        )
        test_probabilities_original = model.predict_proba(
            test_features
        )

    else:
        raise ValueError(model_name)

    cluster_scores = []

    for cluster in range(3):
        mask = train_labels_original == cluster

        cluster_scores.append(
            float(train_thermal[mask].mean())
            if mask.any()
            else float("inf")
        )

    old_components_in_new_order = np.argsort(cluster_scores)
    old_to_new = np.empty(3, dtype=np.int64)

    for new_label, old_label in enumerate(
        old_components_in_new_order
    ):
        old_to_new[old_label] = new_label

    test_labels = old_to_new[test_labels_original]
    test_probabilities = test_probabilities_original[
        :,
        old_components_in_new_order,
    ]

    return test_labels, test_probabilities


def generate_oof_predictions(pm, cache, model_name, seed):
    active = np.asarray(cache["active"])
    folds = np.asarray(cache["fold"])
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    total_rows = len(active)

    raw_labels = np.full(total_rows, -1, dtype=np.int8)
    temporal_labels = np.full(total_rows, -1, dtype=np.int8)
    probabilities = np.full(
        (total_rows, 3),
        np.nan,
        dtype=np.float32,
    )

    for fold in range(5):
        train_indices = np.flatnonzero(
            active & (folds != fold)
        )
        test_indices = np.flatnonzero(
            active & (folds == fold)
        )

        (
            train_features,
            test_features,
            train_thermal,
            _,
        ) = prepare_features(
            pm,
            cache,
            train_indices,
            test_indices,
        )

        labels, fold_probabilities = fit_fold(
            pm,
            model_name,
            seed,
            train_features,
            test_features,
            train_thermal,
        )

        temporal = pm.temporal_decode(
            fold_probabilities,
            video_index[test_indices],
            frame_index[test_indices],
        )

        raw_labels[test_indices] = labels
        temporal_labels[test_indices] = temporal
        probabilities[test_indices] = (
            fold_probabilities.astype(np.float32)
        )

        print(
            f"  {model_name}, seed={seed}, fold={fold}: "
            f"train={len(train_indices)}, test={len(test_indices)}"
        )

    output_path = (
        OUT_DIR
        / f"oof_basic_{model_name}_seed{seed}.npz"
    )

    np.savez(
        output_path,
        raw_label=raw_labels,
        temporal_label=temporal_labels,
        probabilities=probabilities,
    )

    return raw_labels, temporal_labels, probabilities


def regime_statistics(
    labels,
    official,
    active_mask,
    model_name,
    seed,
    label_type,
):
    rows = []

    for regime in range(3):
        mask = active_mask & (labels == regime)

        row = {
            "model": model_name,
            "seed": seed,
            "label_type": label_type,
            "regime": regime,
            "regime_name": [
                "low",
                "nominal",
                "high",
            ][regime],
            "frame_count": int(mask.sum()),
            "frame_fraction": float(
                mask.sum() / max(active_mask.sum(), 1)
            ),
        }

        for column_index, column_name in enumerate(
            OFFICIAL_COLUMNS
        ):
            values = official[mask, column_index]
            values = values[np.isfinite(values)]

            row[f"{column_name}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            row[f"{column_name}_std"] = (
                float(values.std())
                if len(values)
                else np.nan
            )

        rows.append(row)

    return rows


def evaluate_run(
    pm,
    cache,
    official,
    model_name,
    seed,
    raw_labels,
    temporal_labels,
    probabilities,
):
    active = np.asarray(cache["active"])
    video_index = np.asarray(cache["video_index"])
    frame_index = np.asarray(cache["frame_index"])

    power = official[:, 2]
    official_area = official[:, 4]
    official_intensity = official[:, 7]

    valid = (
        active
        & np.isfinite(power)
        & (raw_labels >= 0)
        & np.isfinite(probabilities[:, 2])
    )

    p_high = probabilities[:, 2]

    power_threshold = np.quantile(
        power[valid],
        0.90,
    )
    top_power = valid & (power >= power_threshold)

    raw_sequence = pm.sequence_metrics(
        raw_labels[valid],
        video_index[valid],
        frame_index[valid],
    )
    temporal_sequence = pm.sequence_metrics(
        temporal_labels[valid],
        video_index[valid],
        frame_index[valid],
    )

    per_part_rows = []
    per_part_probability_correlations = []
    per_part_temporal_correlations = []
    per_part_power_differences = []
    monotonic_parts = 0
    eligible_monotonic_parts = 0

    for current_video in np.unique(video_index):
        part_mask = valid & (video_index == current_video)

        if part_mask.sum() < 20:
            continue

        probability_correlation = spearman_correlation(
            p_high[part_mask],
            power[part_mask],
        )
        raw_correlation = spearman_correlation(
            raw_labels[part_mask],
            power[part_mask],
        )
        temporal_correlation = spearman_correlation(
            temporal_labels[part_mask],
            power[part_mask],
        )

        temporal_power_means = []

        for regime in range(3):
            regime_mask = (
                part_mask
                & (temporal_labels == regime)
            )

            temporal_power_means.append(
                float(power[regime_mask].mean())
                if regime_mask.any()
                else np.nan
            )

        monotonic = (
            np.isfinite(temporal_power_means).all()
            and temporal_power_means[0]
            < temporal_power_means[1]
            < temporal_power_means[2]
        )

        if np.isfinite(temporal_power_means).all():
            eligible_monotonic_parts += 1

            difference = (
                temporal_power_means[2]
                - temporal_power_means[0]
            )
            per_part_power_differences.append(difference)

            if monotonic:
                monotonic_parts += 1

        if np.isfinite(probability_correlation):
            per_part_probability_correlations.append(
                probability_correlation
            )

        if np.isfinite(temporal_correlation):
            per_part_temporal_correlations.append(
                temporal_correlation
            )

        per_part_rows.append({
            "model": model_name,
            "seed": seed,
            "video_index": int(current_video),
            "frame_count": int(part_mask.sum()),
            "p_high_power_spearman": probability_correlation,
            "raw_label_power_spearman": raw_correlation,
            "temporal_label_power_spearman": temporal_correlation,
            "low_power_mean": temporal_power_means[0],
            "nominal_power_mean": temporal_power_means[1],
            "high_power_mean": temporal_power_means[2],
            "monotonic_power": bool(monotonic),
        })

    (
        part_corr_mean,
        part_corr_low,
        part_corr_high,
    ) = bootstrap_mean_interval(
        per_part_probability_correlations,
        seed=seed,
    )

    (
        part_temporal_mean,
        part_temporal_low,
        part_temporal_high,
    ) = bootstrap_mean_interval(
        per_part_temporal_correlations,
        seed=seed + 100,
    )

    (
        power_difference_mean,
        power_difference_low,
        power_difference_high,
    ) = bootstrap_mean_interval(
        per_part_power_differences,
        seed=seed + 200,
    )

    result = {
        "model": model_name,
        "seed": seed,
        "active_frames": int(valid.sum()),
        "p_high_power_spearman_frame": spearman_correlation(
            p_high[valid],
            power[valid],
        ),
        "raw_label_power_spearman_frame": spearman_correlation(
            raw_labels[valid],
            power[valid],
        ),
        "temporal_label_power_spearman_frame": (
            spearman_correlation(
                temporal_labels[valid],
                power[valid],
            )
        ),
        "p_high_official_area_spearman": spearman_correlation(
            p_high[valid],
            official_area[valid],
        ),
        "p_high_official_intensity_spearman": (
            spearman_correlation(
                p_high[valid],
                official_intensity[valid],
            )
        ),
        "top_10pct_power_high_regime_recall_raw": float(
            np.mean(raw_labels[top_power] == 2)
        ),
        "top_10pct_power_high_regime_recall_temporal": float(
            np.mean(temporal_labels[top_power] == 2)
        ),
        "part_mean_p_high_power_spearman": part_corr_mean,
        "part_mean_p_high_power_spearman_ci_low": part_corr_low,
        "part_mean_p_high_power_spearman_ci_high": part_corr_high,
        "part_mean_temporal_power_spearman": part_temporal_mean,
        "part_mean_temporal_power_spearman_ci_low": (
            part_temporal_low
        ),
        "part_mean_temporal_power_spearman_ci_high": (
            part_temporal_high
        ),
        "monotonic_parts": monotonic_parts,
        "eligible_monotonic_parts": eligible_monotonic_parts,
        "monotonic_part_fraction": (
            monotonic_parts / eligible_monotonic_parts
            if eligible_monotonic_parts
            else np.nan
        ),
        "part_mean_high_low_power_difference_w": (
            power_difference_mean
        ),
        "part_mean_high_low_power_difference_w_ci_low": (
            power_difference_low
        ),
        "part_mean_high_low_power_difference_w_ci_high": (
            power_difference_high
        ),
        "raw_switches_per_1000": raw_sequence[
            "switches_per_1000"
        ],
        "temporal_switches_per_1000": temporal_sequence[
            "switches_per_1000"
        ],
        "switch_reduction_fraction": (
            1.0
            - temporal_sequence["switches_per_1000"]
            / max(
                raw_sequence["switches_per_1000"],
                1e-12,
            )
        ),
    }

    regime_rows = []

    regime_rows.extend(
        regime_statistics(
            raw_labels,
            official,
            valid,
            model_name,
            seed,
            "raw",
        )
    )
    regime_rows.extend(
        regime_statistics(
            temporal_labels,
            official,
            valid,
            model_name,
            seed,
            "temporal",
        )
    )

    return result, per_part_rows, regime_rows


def alignment_quality(cache, official):
    metadata = cache["metadata"]
    feature_names = metadata["feature_names"]
    features = cache["features"]

    mean_intensity_column = feature_names.index(
        "mean_intensity"
    )
    area_column = feature_names.index(
        "threshold_area"
    )

    our_intensity = np.asarray(
        features[:, mean_intensity_column],
        dtype=np.float64,
    )
    our_area_pixels = np.asarray(
        features[:, area_column],
        dtype=np.float64,
    )

    official_intensity = official[:, 7]
    official_area_pixels = (
        official[:, 4] / PIXEL_AREA_MM2
    )

    intensity_valid = (
        np.isfinite(our_intensity)
        & np.isfinite(official_intensity)
    )

    area_valid = (
        np.isfinite(our_area_pixels)
        & np.isfinite(official_area_pixels)
    )

    return {
        "frame_count": int(len(our_intensity)),
        "intensity_pearson": pearson_correlation(
            our_intensity[intensity_valid],
            official_intensity[intensity_valid],
        ),
        "intensity_spearman": spearman_correlation(
            our_intensity[intensity_valid],
            official_intensity[intensity_valid],
        ),
        "intensity_mae": float(
            np.mean(
                np.abs(
                    our_intensity[intensity_valid]
                    - official_intensity[intensity_valid]
                )
            )
        ),
        "area_spearman": spearman_correlation(
            our_area_pixels[area_valid],
            official_area_pixels[area_valid],
        ),
        "area_pearson": pearson_correlation(
            our_area_pixels[area_valid],
            official_area_pixels[area_valid],
        ),
        "official_area_pixel_conversion": (
            "area_mm2 / 0.000064"
        ),
        "note": (
            "Official area uses a fixed 80-DL threshold; "
            "our area uses adaptive Otsu thresholding, so "
            "correlation rather than exact equality is expected."
        ),
    }


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(run_rows):
    summary_rows = []

    for model_name in MODELS:
        model_rows = [
            row
            for row in run_rows
            if row["model"] == model_name
        ]

        summary = {
            "model": model_name,
            "runs": len(model_rows),
        }

        numeric_fields = [
            key
            for key, value in model_rows[0].items()
            if key not in {"model", "seed"}
            and isinstance(value, (int, float))
        ]

        for field in numeric_fields:
            values = np.asarray(
                [
                    row[field]
                    for row in model_rows
                    if np.isfinite(row[field])
                ],
                dtype=np.float64,
            )

            summary[f"{field}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            summary[f"{field}_std"] = (
                float(values.std())
                if len(values)
                else np.nan
            )

        summary_rows.append(summary)

    return summary_rows


def aggregate_regimes(regime_rows):
    groups = {}

    for row in regime_rows:
        key = (
            row["model"],
            row["label_type"],
            row["regime"],
            row["regime_name"],
        )
        groups.setdefault(key, []).append(row)

    output = []

    for key, rows in sorted(groups.items()):
        result = {
            "model": key[0],
            "label_type": key[1],
            "regime": key[2],
            "regime_name": key[3],
            "runs": len(rows),
        }

        for field in [
            "frame_fraction",
            "power_w_mean",
            "speed_mm_s_mean",
            "area_mm2_mean",
            "length_mm_mean",
            "width_mm_mean",
            "mean_intensity_mean",
            "spatter_count_mean",
            "spatter_area_mean",
        ]:
            values = np.asarray(
                [
                    row[field]
                    for row in rows
                    if np.isfinite(row[field])
                ],
                dtype=np.float64,
            )

            result[f"{field}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            result[f"{field}_std"] = (
                float(values.std())
                if len(values)
                else np.nan
            )

        output.append(result)

    return output


def create_figures(
    cache,
    official,
    gmm_labels,
    gmm_probabilities,
):
    try:
        warnings.filterwarnings(
            "ignore",
            message="Unable to import Axes3D",
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Figures skipped:", exc)
        return

    active = np.asarray(cache["active"])
    video_index = np.asarray(cache["video_index"])
    metadata = cache["metadata"]
    feature_names = metadata["feature_names"]
    features = cache["features"]

    power = official[:, 2]

    valid = (
        active
        & (gmm_labels >= 0)
        & np.isfinite(power)
    )

    power_groups = [
        power[valid & (gmm_labels == regime)]
        for regime in range(3)
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        power_groups,
        labels=["Low", "Nominal", "High"],
        showfliers=False,
    )
    axis.set_ylabel("Measured laser power (W)")
    axis.set_xlabel("Temporally stabilized regime")
    axis.set_title("External physical validation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUT_DIR / "01_measured_power_by_regime.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUT_DIR / "01_measured_power_by_regime.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    p_high = gmm_probabilities[:, 2]
    positions = np.flatnonzero(
        valid & np.isfinite(p_high)
    )

    generator = np.random.default_rng(42)

    if len(positions) > 12000:
        positions = generator.choice(
            positions,
            size=12000,
            replace=False,
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        power[positions],
        p_high[positions],
        s=6,
        alpha=0.25,
    )
    axis.set_xlabel("Measured laser power (W)")
    axis.set_ylabel("Probability of high-thermal regime")
    axis.set_title("Probabilistic regime score")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUT_DIR / "02_high_probability_vs_power.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUT_DIR / "02_high_probability_vs_power.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    intensity_column = feature_names.index(
        "mean_intensity"
    )
    our_intensity = np.asarray(
        features[:, intensity_column]
    )
    official_intensity = official[:, 7]

    positions = np.flatnonzero(
        np.isfinite(our_intensity)
        & np.isfinite(official_intensity)
    )

    if len(positions) > 12000:
        positions = generator.choice(
            positions,
            size=12000,
            replace=False,
        )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        official_intensity[positions],
        our_intensity[positions],
        s=6,
        alpha=0.25,
    )

    limits = [
        min(
            float(official_intensity[positions].min()),
            float(our_intensity[positions].min()),
        ),
        max(
            float(official_intensity[positions].max()),
            float(our_intensity[positions].max()),
        ),
    ]

    axis.plot(limits, limits, linestyle="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Official mean image intensity")
    axis.set_ylabel("Extracted mean image intensity")
    axis.set_title("Frame-level alignment check")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUT_DIR / "03_intensity_alignment.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUT_DIR / "03_intensity_alignment.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_markdown(alignment, summaries, regime_summary):
    path = OUT_DIR / "physical_validation_summary.md"

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PhysMelt physical validation\n\n")

        handle.write("## Synchronization quality\n\n")
        handle.write(
            f"- Intensity Pearson correlation: "
            f"{alignment['intensity_pearson']:.6f}\n"
        )
        handle.write(
            f"- Intensity Spearman correlation: "
            f"{alignment['intensity_spearman']:.6f}\n"
        )
        handle.write(
            f"- Intensity MAE: "
            f"{alignment['intensity_mae']:.6f} DL\n"
        )
        handle.write(
            f"- Area Spearman correlation: "
            f"{alignment['area_spearman']:.4f}\n\n"
        )

        handle.write("## Out-of-fold external validation\n\n")
        handle.write(
            "| Model | P(high)-power Spearman | "
            "Part-balanced Spearman | Monotonic parts | "
            "High-low power gap | Switch reduction |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for row in summaries:
            handle.write(
                f"| {row['model']} "
                f"| {row['p_high_power_spearman_frame_mean']:.3f} "
                f"| {row['part_mean_p_high_power_spearman_mean']:.3f} "
                f"| {100 * row['monotonic_part_fraction_mean']:.1f}% "
                f"| {row['part_mean_high_low_power_difference_w_mean']:.2f} W "
                f"| {100 * row['switch_reduction_fraction_mean']:.1f}% |\n"
            )

        handle.write(
            "\n## GMM temporal regime profiles\n\n"
        )
        handle.write(
            "| Regime | Fraction | Power (W) | Area (mm²) | "
            "Length (mm) | Width (mm) | Intensity |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---:|---:|---:|\n"
        )

        relevant = [
            row
            for row in regime_summary
            if (
                row["model"] == "gmm"
                and row["label_type"] == "temporal"
            )
        ]

        for row in relevant:
            handle.write(
                f"| {row['regime_name']} "
                f"| {100 * row['frame_fraction_mean']:.1f}% "
                f"| {row['power_w_mean_mean']:.2f} "
                f"| {row['area_mm2_mean_mean']:.5f} "
                f"| {row['length_mm_mean_mean']:.4f} "
                f"| {row['width_mm_mean_mean']:.4f} "
                f"| {row['mean_intensity_mean_mean']:.3f} |\n"
            )

    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pm = load_pipeline_module()
    cache = pm.load_cache()

    print("=== NIST SYNCHRONIZED DATA ===")
    official = load_official_tables(cache)

    print()
    print("=== ALIGNMENT QUALITY ===")
    alignment = alignment_quality(cache, official)
    print(json.dumps(alignment, indent=2))

    atomic_json(
        OUT_DIR / "alignment_quality.json",
        alignment,
    )

    run_rows = []
    per_part_rows = []
    regime_rows = []

    saved_gmm_temporal = None
    saved_gmm_probabilities = None

    print()
    print("=== FIVE-FOLD OUT-OF-FOLD MODELS ===")

    for model_name in MODELS:
        for seed in SEEDS:
            print()
            print(
                f"MODEL={model_name}, SEED={seed}"
            )

            (
                raw_labels,
                temporal_labels,
                probabilities,
            ) = generate_oof_predictions(
                pm,
                cache,
                model_name,
                seed,
            )

            (
                result,
                current_per_part,
                current_regimes,
            ) = evaluate_run(
                pm,
                cache,
                official,
                model_name,
                seed,
                raw_labels,
                temporal_labels,
                probabilities,
            )

            run_rows.append(result)
            per_part_rows.extend(current_per_part)
            regime_rows.extend(current_regimes)

            print(json.dumps(result, indent=2))

            if model_name == "gmm" and seed == 42:
                saved_gmm_temporal = temporal_labels
                saved_gmm_probabilities = probabilities

    summaries = aggregate_runs(run_rows)
    regime_summary = aggregate_regimes(regime_rows)

    write_csv(
        OUT_DIR / "external_validation_runs.csv",
        run_rows,
    )
    write_csv(
        OUT_DIR / "external_validation_summary.csv",
        summaries,
    )
    write_csv(
        OUT_DIR / "per_part_external_validation.csv",
        per_part_rows,
    )
    write_csv(
        OUT_DIR / "regime_profiles_all_runs.csv",
        regime_rows,
    )
    write_csv(
        OUT_DIR / "regime_profiles_summary.csv",
        regime_summary,
    )

    atomic_json(
        OUT_DIR / "external_validation_summary.json",
        {
            "alignment": alignment,
            "model_summary": summaries,
            "regime_summary": regime_summary,
        },
    )

    markdown_path = write_markdown(
        alignment,
        summaries,
        regime_summary,
    )

    if (
        saved_gmm_temporal is not None
        and saved_gmm_probabilities is not None
    ):
        create_figures(
            cache,
            official,
            saved_gmm_temporal,
            saved_gmm_probabilities,
        )

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for summary in summaries:
        print()
        print("Model:", summary["model"])
        print(
            "  P(high)-power Spearman:",
            f"{summary['p_high_power_spearman_frame_mean']:.3f}",
        )
        print(
            "  Part-balanced Spearman:",
            f"{summary['part_mean_p_high_power_spearman_mean']:.3f}",
        )
        print(
            "  Monotonic parts:",
            f"{100 * summary['monotonic_part_fraction_mean']:.1f}%",
        )
        print(
            "  High-low power difference:",
            f"{summary['part_mean_high_low_power_difference_w_mean']:.2f} W",
        )
        print(
            "  Temporal switch reduction:",
            f"{100 * summary['switch_reduction_fraction_mean']:.1f}%",
        )

    print()
    print("Markdown summary:", markdown_path)
    print("Output directory:", OUT_DIR)
    print()
    print("Validation finished. The terminal remains open.")


if __name__ == "__main__":
    main()
