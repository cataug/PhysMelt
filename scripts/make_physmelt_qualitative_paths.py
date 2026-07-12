#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import importlib.util
import math
import re
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize

ROOT = Path.home() / "PhysMelt"
PIPELINE = ROOT / "scripts/run_physmelt_full.py"
VALIDATOR = ROOT / "scripts/validate_nist_physics.py"
OOF = ROOT / "outputs/physical_validation/oof_basic_gmm_seed42.npz"

OUT = ROOT / "PLOTS_QUALITATIVE_PATHS"
PDF = OUT / "PDF"
PNG = OUT / "PNG"

FPS = 20000.0
WINDOW = 300

BG = "#F4F6FA"
PANEL = "#FFFFFF"
TEXT = "#1D2635"
GRID = "#DCE2EA"
EDGE = "#C7D0DC"

STATE_NAMES = ["Low thermal", "Nominal", "High thermal"]
STATE_COLORS = ["#5B8DB8", "#59A98D", "#D87459"]
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], STATE_CMAP.N)

POWER_CMAP = LinearSegmentedColormap.from_list(
    "power",
    ["#263A63", "#3E7EA6", "#58B6A4", "#E7C865", "#D8664D"],
)
PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob",
    ["#F6F8FB", "#C4DDEB", "#78B0CC", "#716CB0", "#C04E72"],
)
CORR_CMAP = ListedColormap(["#D9DEE7", "#D95F59"])
CORR_NORM = BoundaryNorm([-0.5, 0.5, 1.5], CORR_CMAP.N)


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": EDGE,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def badge(ax, x, y, text, edge="#56657A", ha="left", va="top"):
    artist = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#FFFFFF",
            "edgecolor": edge,
            "linewidth": 1.3,
            "alpha": 0.97,
        },
        zorder=50,
    )
    artist.set_path_effects([
        pe.withSimplePatchShadow(
            offset=(1.1, -1.1),
            shadow_rgbFace="#C9D0DA",
            alpha=0.45,
        )
    ])


def save(fig, stem):
    PDF.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)

    pdf = PDF / f"{stem}.pdf"
    png = PNG / f"{stem}.png"

    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    print("[SAVED]", pdf)
    print("[SAVED]", png)


def entropy(probabilities):
    p = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / math.log(p.shape[1])


def rolling(values, window, average=True):
    values = np.asarray(values, dtype=float)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float)
    if average:
        kernel /= window
    return np.convolve(values, kernel, mode="valid")


def part_number(video_name):
    match = re.search(r"P(\d+)$", video_name)
    if not match:
        raise ValueError(video_name)
    return int(match.group(1))


def active_positions(cache, pred, video_index):
    positions = np.flatnonzero(cache["video_index"] == video_index)
    positions = positions[np.argsort(cache["frame_index"][positions])]
    valid = (
        cache["active"][positions]
        & (pred["temporal"][positions] >= 0)
        & np.isfinite(pred["prob"][positions, 2])
    )
    return positions[valid]


def build_records(cache, official, pred):
    records = []

    for video_index, video_name in enumerate(cache["metadata"]["video_names"]):
        positions = active_positions(cache, pred, video_index)

        power = official[positions, 2]
        raw = pred["raw"][positions]
        temporal = pred["temporal"][positions]
        uncertainty = entropy(pred["prob"][positions])

        transitions = np.zeros(len(temporal), dtype=float)
        transitions[1:] = temporal[1:] != temporal[:-1]
        corrections = (raw != temporal).astype(float)

        records.append({
            "video_index": video_index,
            "video_name": video_name,
            "part": part_number(video_name),
            "positions": positions,
            "high_power": float(np.max(rolling(power, WINDOW, True))),
            "cleanup": float(np.max(rolling(corrections, WINDOW, False))),
            "transition": float(np.max(rolling(transitions, WINDOW, False))),
            "uncertain": float(np.max(rolling(uncertainty, WINDOW, True))),
            "mean_power": float(np.mean(power)),
            "high_fraction": float(np.mean(temporal == 2)),
        })

    return records


def choose_cases(records):
    used = set()
    selected = []

    for name, key in [
        ("high_power", "high_power"),
        ("temporal_cleanup", "cleanup"),
        ("transition_rich", "transition"),
        ("uncertain_boundary", "uncertain"),
    ]:
        candidates = [row for row in records if row["part"] not in used]
        chosen = max(candidates, key=lambda row: row[key])
        used.add(chosen["part"])
        selected.append((name, chosen))

    return selected


def choose_window(name, record, official, pred):
    positions = record["positions"]

    power = official[positions, 2]
    raw = pred["raw"][positions]
    temporal = pred["temporal"][positions]
    uncertainty = entropy(pred["prob"][positions])
    window = min(WINDOW, len(positions))

    if name == "high_power":
        score = rolling(power, window, True)
    elif name == "temporal_cleanup":
        score = rolling((raw != temporal).astype(float), window, False)
    elif name == "transition_rich":
        transition = np.zeros(len(temporal), dtype=float)
        transition[1:] = temporal[1:] != temporal[:-1]
        enter_high = np.zeros(len(temporal), dtype=float)
        enter_high[1:] = (
            (temporal[1:] == 2)
            & (temporal[:-1] != 2)
        )
        score = rolling(transition + 2.5 * enter_high, window, False)
    else:
        score = rolling(uncertainty, window, True)

    start = int(np.nanargmax(score))
    end = min(start + window - 1, len(positions) - 1)

    return {"start": start, "end": end}


def make_segments(x, y):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def draw_path(ax, x, y, values, cmap, norm):
    ax.plot(
        x,
        y,
        color="#D8DEE7",
        linewidth=5.4,
        alpha=0.78,
        solid_capstyle="round",
        zorder=1,
    )

    collection = LineCollection(
        make_segments(x, y),
        cmap=cmap,
        norm=norm,
        linewidths=3.2,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    collection.set_array(np.asarray(values[:-1]))
    ax.add_collection(collection)

    ax.scatter(
        [x[0]],
        [y[0]],
        marker="^",
        s=70,
        facecolor="white",
        edgecolor=TEXT,
        linewidth=1.3,
        zorder=6,
    )
    ax.scatter(
        [x[-1]],
        [y[-1]],
        marker="s",
        s=58,
        facecolor="white",
        edgecolor=TEXT,
        linewidth=1.3,
        zorder=6,
    )

    xpad = max((np.max(x) - np.min(x)) * 0.04, 0.02)
    ypad = max((np.max(y) - np.min(y)) * 0.04, 0.02)

    ax.set_xlim(np.min(x) - xpad, np.max(x) + xpad)
    ax.set_ylim(np.min(y) - ypad, np.max(y) + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return collection


def highlight(ax, x, y, start, end):
    wx = x[start:end + 1]
    wy = y[start:end + 1]

    ax.plot(
        wx,
        wy,
        color="white",
        linewidth=7.2,
        solid_capstyle="round",
        zorder=7,
    )
    ax.plot(
        wx,
        wy,
        color="#172236",
        linewidth=3.0,
        solid_capstyle="round",
        zorder=8,
    )


def make_atlas(cases, windows, official, pred):
    all_power = np.concatenate([
        official[record["positions"], 2]
        for _, record in cases
    ])
    power_norm = Normalize(
        vmin=float(np.quantile(all_power, 0.02)),
        vmax=float(np.quantile(all_power, 0.98)),
    )

    fig, axes = plt.subplots(
        len(cases),
        4,
        figsize=(20, 18),
        facecolor=BG,
        constrained_layout=True,
    )

    power_map = None
    prob_map = None

    for row, (name, record) in enumerate(cases):
        positions = record["positions"]

        x = official[positions, 0]
        y = official[positions, 1]
        power = official[positions, 2]
        p_high = pred["prob"][positions, 2]
        temporal = pred["temporal"][positions]
        corrected = (
            pred["raw"][positions]
            != pred["temporal"][positions]
        ).astype(float)

        power_map = draw_path(
            axes[row, 0],
            x,
            y,
            power,
            POWER_CMAP,
            power_norm,
        )
        prob_map = draw_path(
            axes[row, 1],
            x,
            y,
            p_high,
            PROB_CMAP,
            Normalize(0, 1),
        )
        draw_path(
            axes[row, 2],
            x,
            y,
            temporal,
            STATE_CMAP,
            STATE_NORM,
        )
        draw_path(
            axes[row, 3],
            x,
            y,
            corrected,
            CORR_CMAP,
            CORR_NORM,
        )

        for column in range(4):
            highlight(
                axes[row, column],
                x,
                y,
                windows[name]["start"],
                windows[name]["end"],
            )

        if row == 0:
            axes[row, 0].set_title("Measured power")
            axes[row, 1].set_title(r"$P(\mathrm{high})$")
            axes[row, 2].set_title("Temporal regime")
            axes[row, 3].set_title("Frames changed by Viterbi")

        badge(
            axes[row, 0],
            0.02,
            0.98,
            (
                f"{name.replace('_', ' ').title()}\n"
                f"P{record['part']:02d} · active indices "
                f"{windows[name]['start']}–{windows[name]['end']}"
            ),
            edge=STATE_COLORS[2],
        )

        badge(
            axes[row, 2],
            0.98,
            0.03,
            (
                f"mean power {record['mean_power']:.1f} W\n"
                f"high state {100 * record['high_fraction']:.1f}%"
            ),
            edge=STATE_COLORS[1],
            ha="right",
            va="bottom",
        )

    fig.suptitle(
        "Laser-path qualitative atlas: measured process, model state, and temporal corrections",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )

    cbar1 = fig.colorbar(
        power_map,
        ax=axes[:, 0].tolist(),
        fraction=0.025,
        pad=0.01,
    )
    cbar1.set_label("Measured power (W)", fontsize=13)

    cbar2 = fig.colorbar(
        prob_map,
        ax=axes[:, 1].tolist(),
        fraction=0.025,
        pad=0.01,
    )
    cbar2.set_label(r"$P(\mathrm{high})$", fontsize=13)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STATE_COLORS[index],
            linewidth=6,
            label=STATE_NAMES[index],
        )
        for index in range(3)
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#D95F59",
            linewidth=6,
            label="Temporal correction",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        edgecolor=EDGE,
        bbox_to_anchor=(0.5, -0.015),
    )

    save(fig, "01_laser_path_qualitative_atlas")


def make_timeline(name, record, window, cache, official, pred):
    positions = record["positions"]
    selected = positions[window["start"]:window["end"] + 1]

    frame_numbers = np.asarray(cache["frame_index"][selected], dtype=float)
    time_ms = frame_numbers / FPS * 1000.0

    power = official[selected, 2]
    area = official[selected, 4]
    intensity = official[selected, 7]
    p_high = pred["prob"][selected, 2]
    raw = pred["raw"][selected]
    temporal = pred["temporal"][selected]
    corrected = raw != temporal

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.5, 10.5),
        facecolor=BG,
        constrained_layout=True,
    )

    ax1, ax2, ax3 = axes

    ax1.plot(
        time_ms,
        power,
        color="#D66A4D",
        linewidth=2.8,
        label="Measured power",
    )
    ax1b = ax1.twinx()
    ax1b.plot(
        time_ms,
        p_high,
        color="#6E64B4",
        linewidth=2.5,
        label=r"$P(\mathrm{high})$",
    )
    ax1b.fill_between(
        time_ms,
        0,
        p_high,
        color="#6E64B4",
        alpha=0.10,
    )
    ax1b.set_ylim(-0.03, 1.03)
    ax1.set_ylabel("Power (W)", color="#D66A4D")
    ax1b.set_ylabel(r"$P(\mathrm{high})$", color="#6E64B4")
    ax1.set_title("Measured laser power vs visual posterior", loc="left")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        ncol=2,
        frameon=True,
        edgecolor=EDGE,
    )

    ax2.plot(
        time_ms,
        area,
        color="#4F9B80",
        linewidth=2.5,
        label="Official area",
    )
    ax2b = ax2.twinx()
    ax2b.plot(
        time_ms,
        intensity,
        color="#627CB5",
        linewidth=2.4,
        label="Official intensity",
    )
    ax2.set_ylabel("Area (mm²)", color="#4F9B80")
    ax2b.set_ylabel("Intensity (DL)", color="#627CB5")
    ax2.set_title("Melt-pool response in the same interval", loc="left")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        ncol=2,
        frameon=True,
        edgecolor=EDGE,
    )

    ax3.step(
        time_ms,
        raw,
        where="post",
        color="#9A9FA8",
        linewidth=2.0,
        linestyle="--",
        label="Raw GMM state",
    )
    ax3.step(
        time_ms,
        temporal,
        where="post",
        color="#1F5F73",
        linewidth=3.0,
        label="Temporal state",
    )
    ax3.fill_between(
        time_ms,
        -0.25,
        2.25,
        where=corrected,
        step="post",
        color="#D76868",
        alpha=0.18,
        label="Corrected frames",
    )
    ax3.set_ylim(-0.35, 2.35)
    ax3.set_yticks([0, 1, 2], ["Low", "Nominal", "High"])
    ax3.set_xlabel("Physical time (ms)")
    ax3.set_ylabel("Regime")
    ax3.set_title("Raw vs temporally stabilized regimes", loc="left")
    ax3.legend(
        loc="upper left",
        ncol=3,
        frameon=True,
        edgecolor=EDGE,
    )

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    correction_fraction = float(np.mean(corrected))

    fig.suptitle(
        (
            f"{name.replace('_', ' ').title()} — "
            f"P{record['part']:02d}, frames "
            f"{int(frame_numbers[0])}–{int(frame_numbers[-1])}"
        ),
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )

    badge(
        ax3,
        0.985,
        0.95,
        f"corrected {100 * correction_fraction:.1f}% of frames",
        edge=STATE_COLORS[1],
        ha="right",
    )

    save(fig, f"timeline_{name}_P{record['part']:02d}")

    return {
        "case": name,
        "part": record["part"],
        "frame_start": int(frame_numbers[0]),
        "frame_end": int(frame_numbers[-1]),
        "mean_power_w": float(np.mean(power)),
        "max_power_w": float(np.max(power)),
        "mean_p_high": float(np.mean(p_high)),
        "corrected_fraction": correction_fraction,
    }


def write_manifest(rows):
    path = OUT / "qualitative_manifest.csv"

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVED]", path)


def main():
    setup_style()

    for directory in [OUT, PDF, PNG]:
        directory.mkdir(parents=True, exist_ok=True)

    for required in [PIPELINE, VALIDATOR, OOF]:
        if not required.exists():
            print("ERROR: missing", required)
            return

    pipeline = import_module("physmelt_pipeline", PIPELINE)
    validator = import_module("physmelt_validator", VALIDATOR)

    cache = pipeline.load_cache()

    # X/Y paths and all physical variables are read from the synchronized
    # 10-column CSV files inside RHF_Analysis_Results.zip.
    official = validator.load_official_tables(cache)

    with np.load(OOF) as npz:
        pred = {
            "raw": np.asarray(npz["raw_label"], dtype=np.int8),
            "temporal": np.asarray(npz["temporal_label"], dtype=np.int8),
            "prob": np.asarray(npz["probabilities"], dtype=np.float64),
        }

    records = build_records(cache, official, pred)
    cases = choose_cases(records)
    windows = {
        name: choose_window(name, record, official, pred)
        for name, record in cases
    }

    print()
    print("Selected qualitative cases:")
    for name, record in cases:
        print(f"  {name:20s} -> P{record['part']:02d}")

    make_atlas(cases, windows, official, pred)

    rows = [
        make_timeline(
            name,
            record,
            windows[name],
            cache,
            official,
            pred,
        )
        for name, record in cases
    ]

    write_manifest(rows)

    print()
    print("DONE")
    print("Output:", OUT)
    print("Terminal remains open.")


if __name__ == "__main__":
    main()
