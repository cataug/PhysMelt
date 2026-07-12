#!/usr/bin/env python3

import json
import math
import warnings
from csv import DictReader
from pathlib import Path

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D",
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path.home() / "PhysMelt"

REPORT_DIR = ROOT / "outputs/reports"
PHYSICAL_DIR = ROOT / "outputs/physical_validation"
FINAL_DIR = ROOT / "outputs/final_evaluation"
CACHE_DIR = ROOT / "data/model_cache"

OUT_DIR = ROOT / "PLOTS_WOW"
PDF_DIR = OUT_DIR / "PDF"
PNG_DIR = OUT_DIR / "PNG"

EXPERIMENT_SUMMARY = REPORT_DIR / "experiment_summary.csv"
STABILITY_CSV = REPORT_DIR / "seed_stability.csv"

EXTERNAL_SUMMARY = (
    PHYSICAL_DIR / "external_validation_summary.csv"
)
REGIME_SUMMARY = (
    PHYSICAL_DIR / "regime_profiles_summary.csv"
)
PER_PART_CSV = (
    PHYSICAL_DIR / "per_part_external_validation.csv"
)

OFFICIAL_NPY = (
    PHYSICAL_DIR / "official_synchronized_data.npy"
)
OOF_GMM = (
    PHYSICAL_DIR / "oof_basic_gmm_seed42.npz"
)
ACTIVE_NPY = CACHE_DIR / "active.npy"

ALARM_CSV = FINAL_DIR / "alarm_evaluation.csv"
EDGE_JSON = FINAL_DIR / "edge_benchmark.json"


# ---------------------------------------------------------------------
# Visual design
# ---------------------------------------------------------------------

BACKGROUND = "#F5F7FB"
PANEL = "#FFFFFF"
TEXT = "#1C2434"
MUTED = "#697386"
GRID = "#D9DEE8"
EDGE = "#C8CFDC"

PALETTE = {
    "blue": "#5470C6",
    "cyan": "#4DB6AC",
    "green": "#67B99A",
    "yellow": "#E9B949",
    "orange": "#E98B5F",
    "red": "#D76868",
    "purple": "#9270CA",
    "pink": "#D985A8",
    "dark": "#364152",
}

MODEL_COLORS = {
    "kmeans": PALETTE["blue"],
    "gmm": PALETTE["orange"],
    "anchored_gmm": PALETTE["purple"],
}

FEATURE_COLORS = {
    "basic": "#89B4D6",
    "appearance": "#91C7B1",
    "physical": "#E7B77D",
    "physical_temporal": "#C69CCF",
}

REGIME_COLORS = [
    "#76A9CF",
    "#79BEA8",
    "#E88964",
]

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "physmelt_density",
    [
        "#F7F8FC",
        "#C8D9F0",
        "#79B8D1",
        "#5470C6",
        "#61428F",
        "#2A1738",
    ],
)


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 23,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "medium",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "axes.edgecolor": EDGE,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "grid.alpha": 0.65,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": BACKGROUND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })


def polish_axis(ax, grid_axis="both"):
    ax.set_facecolor(PANEL)
    ax.grid(True, axis=grid_axis, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)

    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def add_title(ax, title, subtitle=None):
    ax.set_title(
        title,
        loc="left",
        pad=20,
        color=TEXT,
    )

    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13.5,
            color=MUTED,
        )


def badge(
    ax,
    x,
    y,
    text,
    *,
    color=PALETTE["dark"],
    transform=None,
    ha="center",
    va="center",
    fontsize=13,
):
    if transform is None:
        transform = ax.transData

    artist = ax.text(
        x,
        y,
        text,
        transform=transform,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#FFFFFF",
            "edgecolor": color,
            "linewidth": 1.5,
            "alpha": 0.96,
        },
        zorder=20,
    )

    artist.set_path_effects([
        path_effects.withSimplePatchShadow(
            offset=(1.2, -1.2),
            shadow_rgbFace="#D7DCE5",
            alpha=0.45,
        )
    ])

    return artist


def save_figure(fig, stem):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = PDF_DIR / f"{stem}.pdf"
    png_path = PNG_DIR / f"{stem}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.18,
    )

    fig.savefig(
        png_path,
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.18,
    )

    plt.close(fig)

    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def load_csv(path):
    if not path.exists():
        return []

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(DictReader(handle))


def number(row, key, default=np.nan):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def readable_model(name):
    return {
        "kmeans": "KMeans",
        "gmm": "GMM",
        "anchored_gmm": "Anchored GMM",
    }.get(name, name)


def readable_feature(name):
    return {
        "basic": "Area + intensity",
        "appearance": "Appearance",
        "physical": "Physical",
        "physical_temporal": "Physical + temporal",
    }.get(name, name)


def parse_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


# ---------------------------------------------------------------------
# Figure 1: Representation ablation
# ---------------------------------------------------------------------

def plot_representation_ablation():
    rows = [
        row
        for row in load_csv(EXPERIMENT_SUMMARY)
        if row.get("protocol") == "holdout"
    ]

    if not rows:
        print("[SKIP] Representation ablation: no summary CSV")
        return

    feature_order = [
        "basic",
        "appearance",
        "physical",
        "physical_temporal",
    ]

    model_order = [
        "kmeans",
        "gmm",
        "anchored_gmm",
    ]

    x = np.arange(len(feature_order))
    width = 0.23

    fig, ax = plt.subplots(figsize=(13.5, 8.0))

    for model_index, model_name in enumerate(model_order):
        model_rows = {
            row["feature_set"]: row
            for row in rows
            if row.get("model") == model_name
        }

        values = [
            number(
                model_rows.get(feature_name, {}),
                "silhouette_mean",
            )
            for feature_name in feature_order
        ]

        offset = (
            model_index
            - (len(model_order) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model_name],
            edgecolor="#FFFFFF",
            linewidth=1.8,
            label=readable_model(model_name),
            zorder=4,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.011,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=TEXT,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "#FFFFFF",
                    "edgecolor": MODEL_COLORS[model_name],
                    "linewidth": 1.25,
                    "alpha": 0.97,
                },
                zorder=10,
            )

    # Ненавязчиво выделяем лучший набор признаков.
    ax.axvspan(
        -0.48,
        0.48,
        color=FEATURE_COLORS["basic"],
        alpha=0.12,
        zorder=0,
    )

    ax.set_xticks(
        x,
        [
            readable_feature(feature_name)
            for feature_name in feature_order
        ],
    )

    ax.set_ylabel("Silhouette score ↑")
    ax.set_ylim(0.16, 0.57)

    add_title(
        ax,
        "Minimal physical representation wins",
        (
            "Area and mean intensity preserve clearer cross-part regimes "
            "than higher-dimensional descriptors"
        ),
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        edgecolor=EDGE,
        ncol=3,
    )

    badge(
        ax,
        0.018,
        0.93,
        "Best representation\nArea + intensity",
        transform=ax.transAxes,
        color=PALETTE["blue"],
        ha="left",
        va="top",
        fontsize=13,
    )

    polish_axis(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        "01_representation_ablation",
    )

# ---------------------------------------------------------------------
# Figure 2: Ranked model comparison
# ---------------------------------------------------------------------

def plot_model_ranking():
    rows = [
        row
        for row in load_csv(EXPERIMENT_SUMMARY)
        if row.get("protocol") == "holdout"
    ]

    if not rows:
        print("[SKIP] Model ranking")
        return

    records = []

    for row in rows:
        records.append({
            "label": (
                f"{readable_feature(row['feature_set'])}"
                f" · {readable_model(row['model'])}"
            ),
            "feature": row["feature_set"],
            "value": number(row, "silhouette_mean"),
        })

    records.sort(
        key=lambda item: item["value"]
    )

    fig, ax = plt.subplots(figsize=(13.5, 9.5))

    y = np.arange(len(records))
    values = np.asarray([
        item["value"]
        for item in records
    ])

    for position, item in enumerate(records):
        color = FEATURE_COLORS.get(
            item["feature"],
            PALETTE["blue"],
        )

        ax.hlines(
            position,
            0,
            item["value"],
            linewidth=4,
            color=color,
            alpha=0.55,
            zorder=2,
        )

        ax.scatter(
            item["value"],
            position,
            s=210,
            color=color,
            edgecolor="white",
            linewidth=2.2,
            zorder=5,
        )

        ax.text(
            item["value"] + 0.008,
            position,
            f"{item['value']:.3f}",
            va="center",
            ha="left",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFFFFF",
                "edgecolor": color,
                "alpha": 0.92,
            },
        )

    ax.set_yticks(
        y,
        [item["label"] for item in records],
    )

    ax.set_xlabel("Silhouette score ↑")
    ax.set_xlim(0, max(values) + 0.085)

    ax.axvspan(
        0.45,
        max(values) + 0.085,
        color="#DDF1E6",
        alpha=0.50,
        zorder=0,
    )

    add_title(
        ax,
        "Compact models dominate the full ranking",
        "Holdout evaluation with parts separated across train and test",
    )

    badge(
        ax,
        0.985,
        0.03,
        "Green zone: silhouette ≥ 0.45",
        transform=ax.transAxes,
        color=PALETTE["green"],
        ha="right",
        va="bottom",
    )

    polish_axis(ax, grid_axis="x")
    fig.tight_layout()

    save_figure(
        fig,
        "02_model_ranking",
    )


# ---------------------------------------------------------------------
# Figure 3: External physical validation
# ---------------------------------------------------------------------

def plot_external_validation():
    rows = load_csv(EXTERNAL_SUMMARY)

    if not rows:
        print("[SKIP] External validation")
        return

    rows = [
        row
        for row in rows
        if row.get("model") in {"kmeans", "gmm"}
    ]

    metrics = [
        (
            "Frame-level\nSpearman",
            "p_high_power_spearman_frame_mean",
        ),
        (
            "Part-balanced\nSpearman",
            "part_mean_p_high_power_spearman_mean",
        ),
        (
            "Temporal-state\nSpearman",
            "part_mean_temporal_power_spearman_mean",
        ),
    ]

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 7.4))

    for model_index, model_name in enumerate([
        "kmeans",
        "gmm",
    ]):
        row = next(
            item
            for item in rows
            if item["model"] == model_name
        )

        values = [
            number(row, key)
            for _, key in metrics
        ]

        offset = (
            model_index - 0.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=1.7,
            label=readable_model(model_name),
            zorder=4,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.016,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                color=TEXT,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": MODEL_COLORS[model_name],
                    "alpha": 0.94,
                },
            )

    ax.set_xticks(
        x,
        [label for label, _ in metrics],
    )
    ax.set_ylabel("Spearman correlation with measured power ↑")
    ax.set_ylim(0.0, 0.72)

    add_title(
        ax,
        "Visual regimes align with an unseen physical signal",
        (
            "Measured laser power is used only for external validation, "
            "not for regime discovery"
        ),
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor=EDGE,
    )

    badge(
        ax,
        0.02,
        0.92,
        "Positive alignment\nacross all 55 parts",
        transform=ax.transAxes,
        color=PALETTE["green"],
        ha="left",
        va="top",
    )

    polish_axis(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        "03_external_power_validation",
    )


# ---------------------------------------------------------------------
# Figure 4: Physical reliability scorecard
# ---------------------------------------------------------------------

def plot_reliability_scorecard():
    rows = load_csv(EXTERNAL_SUMMARY)

    if not rows:
        print("[SKIP] Reliability scorecard")
        return

    rows = [
        row
        for row in rows
        if row.get("model") in {"kmeans", "gmm"}
    ]

    labels = [
        "Monotonic parts",
        "Temporal switch reduction",
        "Top-power recall",
    ]

    keys = [
        "monotonic_part_fraction_mean",
        "switch_reduction_fraction_mean",
        "top_10pct_power_high_regime_recall_temporal_mean",
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 7.3))

    for model_index, model_name in enumerate([
        "kmeans",
        "gmm",
    ]):
        row = next(
            item
            for item in rows
            if item["model"] == model_name
        )

        values = [
            100.0 * number(row, key)
            for key in keys
        ]

        offset = (
            model_index - 0.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=1.7,
            label=readable_model(model_name),
            zorder=4,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                color=TEXT,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": MODEL_COLORS[model_name],
                    "alpha": 0.95,
                },
            )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Performance (%) ↑")
    ax.set_ylim(0, 112)

    add_title(
        ax,
        "Reliable regimes without defect labels",
        (
            "GMM preserves monotonic power ordering in all 55 parts; "
            "temporal decoding removes short-lived state flicker"
        ),
    )

    ax.legend(
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor=EDGE,
    )

    polish_axis(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        "04_reliability_scorecard",
    )


# ---------------------------------------------------------------------
# Figure 5: Regime profile heatmap
# ---------------------------------------------------------------------

def plot_regime_heatmap():
    rows = [
        row
        for row in load_csv(REGIME_SUMMARY)
        if (
            row.get("model") == "gmm"
            and row.get("label_type") == "temporal"
        )
    ]

    if len(rows) != 3:
        print("[SKIP] Regime heatmap")
        return

    rows.sort(
        key=lambda row: int(row["regime"])
    )

    metric_specs = [
        ("Power", "power_w_mean_mean", "{:.1f} W"),
        ("Area", "area_mm2_mean_mean", "{:.5f} mm²"),
        ("Length", "length_mm_mean_mean", "{:.4f} mm"),
        ("Width", "width_mm_mean_mean", "{:.4f} mm"),
        ("Intensity", "mean_intensity_mean_mean", "{:.2f} DL"),
    ]

    actual = np.array([
        [
            number(row, key)
            for _, key, _ in metric_specs
        ]
        for row in rows
    ])

    normalized = np.zeros_like(actual)

    for column in range(actual.shape[1]):
        minimum = actual[:, column].min()
        maximum = actual[:, column].max()

        normalized[:, column] = (
            actual[:, column] - minimum
        ) / max(maximum - minimum, 1e-12)

    fig, ax = plt.subplots(figsize=(13.5, 6.6))

    image = ax.imshow(
        normalized,
        cmap=CUSTOM_CMAP,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(len(metric_specs)),
        [name for name, _, _ in metric_specs],
    )

    ax.set_yticks(
        np.arange(3),
        ["Low thermal", "Nominal", "High thermal"],
    )

    for row_index in range(3):
        for column_index, (_, _, formatter) in enumerate(
            metric_specs
        ):
            value = actual[row_index, column_index]

            text_color = (
                "white"
                if normalized[row_index, column_index] > 0.58
                else TEXT
            )

            text = ax.text(
                column_index,
                row_index,
                formatter.format(value),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_color,
            )

            text.set_path_effects([
                path_effects.withStroke(
                    linewidth=2.2,
                    foreground=(
                        "#293141"
                        if text_color == "white"
                        else "#FFFFFF"
                    ),
                )
            ])

    add_title(
        ax,
        "Every physical quantity rises across the discovered regimes",
        "Cell color shows the relative level within each measurement",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label(
        "Relative level within metric",
        fontsize=15,
    )
    colorbar.ax.tick_params(labelsize=12)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "05_regime_physical_profile",
    )


# ---------------------------------------------------------------------
# Figure 6: Seed stability
# ---------------------------------------------------------------------

def plot_seed_stability():
    rows = [
        row
        for row in load_csv(STABILITY_CSV)
        if row.get("protocol") == "holdout"
    ]

    if not rows:
        print("[SKIP] Seed stability")
        return

    feature_order = {
        "basic": 0,
        "appearance": 1,
        "physical": 2,
        "physical_temporal": 3,
    }

    model_order = {
        "kmeans": 0,
        "gmm": 1,
        "anchored_gmm": 2,
    }

    rows.sort(
        key=lambda row: (
            feature_order.get(
                row.get("feature_set"),
                99,
            ),
            model_order.get(
                row.get("model"),
                99,
            ),
        )
    )

    values = np.asarray([
        [
            number(row, "raw_seed_ari"),
            number(row, "temporal_seed_ari"),
        ]
        for row in rows
    ])

    labels = [
        (
            f"{readable_feature(row['feature_set'])}"
            f" · {readable_model(row['model'])}"
        )
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10.5, 10.0))

    image = ax.imshow(
        values,
        cmap=LinearSegmentedColormap.from_list(
            "stability",
            [
                "#F4D8D8",
                "#F4E3B4",
                "#D7EEDC",
                "#65B891",
            ],
        ),
        vmin=0.75,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_xticks(
        [0, 1],
        ["Frame-level ARI", "Temporal ARI"],
    )
    ax.set_yticks(
        np.arange(len(labels)),
        labels,
    )

    for row_index in range(values.shape[0]):
        for column_index in range(2):
            value = values[row_index, column_index]

            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=(
                    "white"
                    if value >= 0.94
                    else TEXT
                ),
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": (
                        "#3E8F70"
                        if value >= 0.94
                        else "#FFFFFF"
                    ),
                    "edgecolor": "none",
                    "alpha": 0.80,
                },
            )

    add_title(
        ax,
        "Seed stability is nearly deterministic",
        "Adjusted Rand Index across seeds 42, 43, and 44",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label("Adjusted Rand Index ↑")
    colorbar.ax.tick_params(labelsize=12)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "06_seed_stability",
    )


# ---------------------------------------------------------------------
# Figure 7: Part-wise high-low power gap
# ---------------------------------------------------------------------

def plot_partwise_power_gap():
    rows = [
        row
        for row in load_csv(PER_PART_CSV)
        if (
            row.get("model") == "gmm"
            and row.get("seed") == "42"
        )
    ]

    if not rows:
        print("[SKIP] Part-wise power gaps")
        return

    records = []

    for row in rows:
        low = number(row, "low_power_mean")
        high = number(row, "high_power_mean")

        if not np.isfinite(low) or not np.isfinite(high):
            continue

        records.append({
            "video_index": int(float(row["video_index"])),
            "gap": high - low,
            "monotonic": parse_bool(
                row.get("monotonic_power")
            ),
        })

    records.sort(
        key=lambda item: item["gap"]
    )

    x = np.arange(len(records))
    gaps = np.asarray([
        item["gap"]
        for item in records
    ])

    colors = [
        PALETTE["green"]
        if item["monotonic"]
        else PALETTE["red"]
        for item in records
    ]

    fig, ax = plt.subplots(figsize=(14.5, 7.2))

    ax.bar(
        x,
        gaps,
        color=colors,
        width=0.82,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )

    mean_gap = float(gaps.mean())

    ax.axhline(
        mean_gap,
        color=PALETTE["dark"],
        linestyle="--",
        linewidth=2.2,
        zorder=6,
    )

    ax.set_xlabel("Parts sorted by high–low power gap")
    ax.set_ylabel("Measured high–low power gap (W) ↑")

    ax.set_xticks(
        np.arange(0, len(records), 5),
        [
            f"P{records[index]['video_index'] + 1:02d}"
            for index in range(
                0,
                len(records),
                5,
            )
        ],
    )

    add_title(
        ax,
        "Thermal-state separation persists across individual parts",
        "GMM out-of-fold predictions, seed 42",
    )

    badge(
        ax,
        0.985,
        0.93,
        f"Mean gap\n{mean_gap:.1f} W",
        transform=ax.transAxes,
        color=PALETTE["dark"],
        ha="right",
        va="top",
    )

    badge(
        ax,
        0.015,
        0.93,
        "55 / 55 parts\nmonotonic",
        transform=ax.transAxes,
        color=PALETTE["green"],
        ha="left",
        va="top",
    )

    polish_axis(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        "07_partwise_power_gap",
    )


# ---------------------------------------------------------------------
# Figure 8: Probability vs measured power density
# ---------------------------------------------------------------------

def plot_probability_power_density():
    required = [
        OFFICIAL_NPY,
        OOF_GMM,
        ACTIVE_NPY,
    ]

    if not all(path.exists() for path in required):
        print("[SKIP] Probability-power density")
        return

    official = np.load(
        OFFICIAL_NPY,
        mmap_mode="r",
    )

    prediction = np.load(OOF_GMM)

    probabilities = np.asarray(
        prediction["probabilities"],
        dtype=np.float64,
    )

    active = np.asarray(
        np.load(
            ACTIVE_NPY,
            mmap_mode="r",
        ),
        dtype=bool,
    )

    power = np.asarray(
        official[:, 2],
        dtype=np.float64,
    )

    p_high = probabilities[:, 2]

    valid = (
        active
        & np.isfinite(power)
        & np.isfinite(p_high)
    )

    summary_rows = load_csv(EXTERNAL_SUMMARY)

    gmm_summary = next(
        (
            row
            for row in summary_rows
            if row.get("model") == "gmm"
        ),
        {},
    )

    rho = number(
        gmm_summary,
        "p_high_power_spearman_frame_mean",
    )

    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    density = ax.hexbin(
        power[valid],
        p_high[valid],
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap=CUSTOM_CMAP,
        linewidths=0.15,
        edgecolors="#FFFFFF",
    )

    ax.axvline(
        180,
        linestyle="--",
        linewidth=2.0,
        color=PALETTE["red"],
        alpha=0.9,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=2.0,
        color=PALETTE["dark"],
        alpha=0.9,
    )

    ax.set_xlabel("Measured laser power (W)")
    ax.set_ylabel(r"$P(\mathrm{high\ thermal})$")
    ax.set_ylim(-0.03, 1.03)

    add_title(
        ax,
        "A visual posterior tracks measured laser power",
        "Out-of-fold GMM predictions over 80,245 active frames",
    )

    badge(
        ax,
        0.03,
        0.94,
        rf"Spearman $\rho$ = {rho:.3f}",
        transform=ax.transAxes,
        color=PALETTE["purple"],
        ha="left",
        va="top",
        fontsize=15,
    )

    ax.text(
        181.5,
        0.05,
        "180 W",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["red"],
    )

    colorbar = fig.colorbar(
        density,
        ax=ax,
        fraction=0.04,
        pad=0.025,
    )

    colorbar.set_label(
        "Frame density, log scale",
        fontsize=15,
    )
    colorbar.ax.tick_params(labelsize=12)

    polish_axis(ax)
    fig.tight_layout()

    save_figure(
        fig,
        "08_probability_power_density",
    )


# ---------------------------------------------------------------------
# Figure 9: Alarm trade-off
# ---------------------------------------------------------------------

def extract_probability_threshold(method):
    prefix = "gmm_probability_"

    if not method.startswith(prefix):
        return None

    value = method[len(prefix):]

    if value == "best_f1_descriptive":
        return None

    try:
        return float(value)
    except Exception:
        return None


def plot_alarm_tradeoff():
    rows = load_csv(ALARM_CSV)

    if not rows:
        print(
            "[SKIP] Alarm trade-off: "
            "final evaluation has not been run"
        )
        return

    definitions = sorted({
        row["alarm_definition"]
        for row in rows
    })

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    definition_colors = [
        PALETTE["purple"],
        PALETTE["red"],
        PALETTE["blue"],
    ]

    for definition_index, definition in enumerate(
        definitions
    ):
        selected = []

        for row in rows:
            if row.get("alarm_definition") != definition:
                continue

            threshold = extract_probability_threshold(
                row.get("method", "")
            )

            if threshold is None:
                continue

            selected.append({
                "threshold": threshold,
                "precision": number(row, "precision"),
                "recall": number(row, "recall"),
                "f1": number(row, "f1"),
            })

        selected.sort(
            key=lambda item: item["threshold"]
        )

        if not selected:
            continue

        color = definition_colors[
            definition_index
            % len(definition_colors)
        ]

        recall = [
            item["recall"]
            for item in selected
        ]
        precision = [
            item["precision"]
            for item in selected
        ]

        ax.plot(
            recall,
            precision,
            marker="o",
            markersize=10,
            linewidth=3,
            color=color,
            label=definition.replace("_", " "),
            zorder=4,
        )

        for item in selected:
            ax.scatter(
                item["recall"],
                item["precision"],
                s=140,
                color=color,
                edgecolor="white",
                linewidth=2,
                zorder=6,
            )

            ax.annotate(
                f"τ={item['threshold']:.2f}",
                xy=(
                    item["recall"],
                    item["precision"],
                ),
                xytext=(7, 8),
                textcoords="offset points",
                fontsize=11.5,
                fontweight="bold",
                color=TEXT,
                bbox={
                    "boxstyle": "round,pad=0.23",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.92,
                },
            )

    ax.set_xlabel("Recall ↑")
    ax.set_ylabel("Precision ↑")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)

    add_title(
        ax,
        "Alarm sensitivity can be tuned without retraining",
        "Fixed GMM probability thresholds define different operating points",
    )

    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor=EDGE,
    )

    polish_axis(ax)
    fig.tight_layout()

    save_figure(
        fig,
        "09_alarm_precision_recall",
    )


# ---------------------------------------------------------------------
# Figure 10: Edge latency
# ---------------------------------------------------------------------

def plot_edge_latency():
    if not EDGE_JSON.exists():
        print(
            "[SKIP] Edge latency: "
            "final evaluation has not been run"
        )
        return

    with EDGE_JSON.open(
        encoding="utf-8"
    ) as handle:
        data = json.load(handle)

    entries = [
        (
            "Raw AVI decode\n+ two features",
            (
                data[
                    "raw_decode_and_minimal_features"
                ]["ms_per_frame"]
                * 1000.0
            ),
            PALETTE["purple"],
        ),
        (
            "Scaling + GMM",
            data[
                "gmm_scaling_plus_inference"
            ]["median_us_per_frame"],
            PALETTE["orange"],
        ),
        (
            "GMM inference",
            data[
                "gmm_inference"
            ]["median_us_per_frame"],
            PALETTE["blue"],
        ),
        (
            "Distilled logistic",
            data[
                "logistic_inference"
            ]["median_us_per_frame"],
            PALETTE["green"],
        ),
    ]

    labels = [
        entry[0]
        for entry in entries
    ]
    values = np.asarray([
        entry[1]
        for entry in entries
    ])
    colors = [
        entry[2]
        for entry in entries
    ]

    y = np.arange(len(entries))

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.8,
        height=0.63,
        zorder=4,
    )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Latency per frame (µs, logarithmic scale) ↓")

    for bar, value, color in zip(
        bars,
        values,
        colors,
    ):
        if value >= 1000:
            label = f"{value / 1000.0:.2f} ms"
        else:
            label = f"{value:.3f} µs"

        ax.text(
            value * 1.16,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.95,
            },
        )

    add_title(
        ax,
        "The probabilistic state model is effectively free",
        (
            "End-to-end throughput is dominated by AVI decoding "
            "and frame-level feature extraction"
        ),
    )

    parameter_count = data.get(
        "gmm_parameter_count",
        0,
    )
    model_bytes = data.get(
        "gmm_model_size_bytes",
        0,
    )

    badge(
        ax,
        0.985,
        0.94,
        (
            f"{parameter_count} numeric parameters\n"
            f"{model_bytes / 1024.0:.1f} KiB serialized"
        ),
        transform=ax.transAxes,
        color=PALETTE["blue"],
        ha="right",
        va="top",
    )

    polish_axis(ax, grid_axis="x")
    fig.tight_layout()

    save_figure(
        fig,
        "10_edge_latency",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    configure_style()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("PhysMelt WOW plot generator")
    print("=" * 78)
    print("Output:", OUT_DIR)
    print()

    plot_representation_ablation()
    plot_model_ranking()
    plot_external_validation()
    plot_reliability_scorecard()
    plot_regime_heatmap()
    plot_seed_stability()
    plot_partwise_power_gap()
    plot_probability_power_density()
    plot_alarm_tradeoff()
    plot_edge_latency()

    generated_pdfs = sorted(
        PDF_DIR.glob("*.pdf")
    )
    generated_pngs = sorted(
        PNG_DIR.glob("*.png")
    )

    manifest = {
        "pdf_count": len(generated_pdfs),
        "png_count": len(generated_pngs),
        "pdf_files": [
            str(path)
            for path in generated_pdfs
        ],
        "png_files": [
            str(path)
            for path in generated_pngs
        ],
    }

    with (
        OUT_DIR / "plot_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("PDF figures:", len(generated_pdfs))
    print("PNG figures:", len(generated_pngs))
    print("Manifest:", OUT_DIR / "plot_manifest.json")
    print()
    print("Terminal remains open.")


if __name__ == "__main__":
    main()
