#!/usr/bin/env python3
"""Plot quantum advantage curves and difficulty (s·κ) histograms from benchmark data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

GATE_SPEED_RECORD = 8e-10  # seconds — update as needed
N_POINTS = 500
N_BINS = 30

CLASS_LABELS = {
    "independent_set": "Independent Set",
    "clique":          "Clique",
    "vertex_cover":    "Vertex Cover",
    "max_flow":        "Max Flow",
    "netlib":          "Netlib",
    "miplib":          "MIPlib",
    "stochlp":         "StochLP",
    "misc":            "Misc",
}
CLASS_COLORS = {
    "independent_set": "#E8A87C",
    "clique":          "#6B8FA8",
    "vertex_cover":    "#7AAA7A",
    "max_flow":        "#C97B7B",
    "netlib":          "#9080B8",
    "miplib":          "#A0A0A0",
    "stochlp":         "#E8A8C8",
    "misc":            "#B5A882",
}

RUNTIME_KEYS = {
    "glpk":      "runtime_glpk",
    "highs-std": "runtime_highs_std",
    "highs-mps": "runtime_highs_mps",
}

# Maps CLI mode names to benchmark data key suffixes
_VARIANT_SUFFIX = {"mnes": "qipm1", "oss": "qipm2"}

_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def _iter_records(instance_classes: list[str], cache_dir: Path) -> dict[str, list[dict]]:
    """Load all .data JSON files, grouped by class."""
    result: dict[str, list[dict]] = {}
    for cls in instance_classes:
        cls_dir = cache_dir / cls
        if not cls_dir.is_dir():
            continue
        records = []
        for instance_dir in sorted(cls_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            data_path = instance_dir / (instance_dir.name + ".data")
            if not data_path.exists():
                continue
            try:
                records.append(json.loads(data_path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        if records:
            result[cls] = records
    return result


# ---------------------------------------------------------------------------
# Advantage plot
# ---------------------------------------------------------------------------

def _load_advantage_data(
    instance_classes: list[str],
    cache_dir: Path,
    runtime_key: str,
) -> dict[str, list[dict]]:
    """Filter records to those with a valid runtime and at least one gate count."""
    all_records = _iter_records(instance_classes, cache_dir)
    result = {}
    for cls, records in all_records.items():
        filtered = [
            r for r in records
            if r.get(runtime_key)
            and (r.get("gate_count_qipm1") is not None or r.get("gate_count_qipm2") is not None)
        ]
        if filtered:
            result[cls] = filtered
    return result


def _gate_counts(records: list[dict], variant: str) -> np.ndarray | None:
    """Extract gate counts for a single variant ('mnes' or 'oss')."""
    key = "gate_count_" + _VARIANT_SUFFIX[variant]
    vals = [r[key] for r in records if r.get(key) is not None]
    return np.array(vals, dtype=np.float64) if vals else None


def _crossover_times(gate_counts: np.ndarray, runtimes: np.ndarray) -> np.ndarray:
    return runtimes / gate_counts


def _advantage_curve(ct: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    return np.array([100.0 * np.mean(t < ct) for t in t_values])


def _truncate_at_zero(
    t_values: np.ndarray, curve: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    zero_idx = np.argmax(curve == 0.0)
    if curve[zero_idx] == 0.0:
        return t_values[: zero_idx + 1], curve[: zero_idx + 1]
    return t_values, curve


def plot_advantage(
    instance_classes: list[str],
    mode: str,
    cache_dir: Path,
    output: Path,
    runtime_key: str = "runtime_glpk",
) -> None:
    """Plot advantage curves. mode: 'mnes', 'oss', or 'both'."""
    data = _load_advantage_data(instance_classes, cache_dir, runtime_key)
    if not data:
        print("No data found.")
        return

    variants = list(_VARIANT_SUFFIX) if mode == "both" else [mode]

    all_cts: list[np.ndarray] = []
    for cls, records in data.items():
        for variant in variants:
            gc = _gate_counts(records, variant)
            if gc is None:
                continue
            key = "gate_count_" + _VARIANT_SUFFIX[variant]
            hrs = np.array([r[runtime_key] for r in records if r.get(key) is not None], dtype=np.float64)
            if len(gc) == len(hrs) and len(gc) > 0:
                all_cts.append(_crossover_times(gc, hrs))

    if not all_cts:
        print("No valid data for the requested mode.")
        return

    combined = np.concatenate(all_cts)
    x_min = float(combined.min())
    x_max = max(float(combined.max()), GATE_SPEED_RECORD) * 10
    t_values = np.geomspace(x_min, x_max, N_POINTS)

    plt.rcParams.update({
        **_RCPARAMS,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.8,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    linestyles = {"mnes": "-", "oss": "--"}

    for i, (cls, records) in enumerate(data.items()):
        color = CLASS_COLORS.get(cls, fallback_colors[i % len(fallback_colors)])
        for variant in variants:
            key = "gate_count_" + _VARIANT_SUFFIX[variant]
            pairs = [(r, r[runtime_key]) for r in records if r.get(key) is not None]
            if not pairs:
                continue
            gc = np.array([r[key] for r, _ in pairs], dtype=np.float64)
            hr = np.array([h for _, h in pairs], dtype=np.float64)
            ct = _crossover_times(gc, hr)
            curve = _advantage_curve(ct, t_values)
            tv, cv = _truncate_at_zero(t_values, curve)
            ax.plot(tv, cv, color=color, linestyle=linestyles[variant], linewidth=1.8)

    ax.axvline(GATE_SPEED_RECORD, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(
        GATE_SPEED_RECORD * 0.82, 50,
        "current speed record for\n an entangling gate operation",
        ha="right", va="center", fontsize=8.5, rotation=90, color="#444444",
    )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("gate execution time ($s$)", fontsize=11, labelpad=8)
    ax.set_ylabel(r"instances with quantum advantage (\%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=9.5)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    class_handles = [
        mpatches.Patch(
            facecolor=CLASS_COLORS.get(cls, fallback_colors[i % len(fallback_colors)]),
            edgecolor="none",
            label=CLASS_LABELS.get(cls, cls),
        )
        for i, cls in enumerate(data)
    ]
    fig.legend(
        handles=class_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(class_handles), 4),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=9,
    )

    if mode == "both":
        h1 = mlines.Line2D([], [], color="black", linestyle="-",  linewidth=1.8, label="QIPM (MNES)")
        h2 = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.8, label="QIPM (OSS)")
        ax.legend(handles=[h1, h2], loc="lower left", fontsize=9,
                  framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# Difficulty plot
# ---------------------------------------------------------------------------

def _load_difficulty_data(
    instance_classes: list[str],
    cache_dir: Path,
    variant: str,
) -> dict[str, np.ndarray]:
    """Load s·κ products for each class for the given variant ('mnes' or 'oss')."""
    suffix = _VARIANT_SUFFIX[variant]
    sparsity_key = f"sparsity_{suffix}"
    cond_key = f"cond_{suffix}"
    all_records = _iter_records(instance_classes, cache_dir)
    result: dict[str, np.ndarray] = {}
    for cls, records in all_records.items():
        values = []
        for r in records:
            s = r.get(sparsity_key)
            k = r.get(cond_key)
            if s is None or k is None:
                continue
            values.append(float(s) * float(k))
        if values:
            result[cls] = np.array(values, dtype=np.float64)
    return result


def plot_difficulty(
    instance_classes: list[str],
    variant: str,
    cache_dir: Path,
    output: Path,
) -> None:
    """Plot stacked s·κ histogram for one variant ('mnes' or 'oss')."""
    data = _load_difficulty_data(instance_classes, cache_dir, variant)
    if not data:
        print(f"No difficulty data found for {variant}; skipping.")
        return

    all_values = np.concatenate(list(data.values()))
    pos = all_values[all_values > 0]
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), N_BINS + 1)

    plt.rcParams.update(_RCPARAMS)

    fig, ax = plt.subplots(figsize=(8, 4))

    classes_sorted = sorted(data.keys(), key=lambda c: float(np.median(data[c])))
    ax.hist(
        [data[cls] for cls in classes_sorted],
        bins=bins,
        stacked=True,
        color=[CLASS_COLORS.get(cls, "#888888") for cls in classes_sorted],
        label=[CLASS_LABELS.get(cls, cls) for cls in classes_sorted],
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$s \cdot \kappa$", fontsize=11, labelpad=8)
    ax.set_ylabel("number of instances", fontsize=11, labelpad=8)
    ax.set_title(variant.upper(), fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot quantum advantage curves or difficulty (s·κ) histograms.",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--mode",
        choices=["mnes", "oss", "both"],
        default="both",
        help="Which QIPM variant(s) to include (default: both).",
    )
    parser.add_argument(
        "--solver",
        choices=list(RUNTIME_KEYS),
        default="glpk",
        help="Classical solver runtime to compare against (default: glpk). Ignored with --difficulty.",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Plot s·κ difficulty histogram instead of advantage curves.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir if args.cache_dir is not None else Path("cache_dir").resolve()

    if args.instance_classes:
        classes = args.instance_classes
        classes_tag = "-".join(classes)
    else:
        classes = [d.name for d in sorted(cache_dir.iterdir()) if d.is_dir()] if cache_dir.is_dir() else []
        classes_tag = "all"

    if args.difficulty:
        variants = list(_VARIANT_SUFFIX) if args.mode == "both" else [args.mode]
        for variant in variants:
            plot_difficulty(
                instance_classes=classes,
                variant=variant,
                cache_dir=cache_dir,
                output=Path(f"plot_difficulty_{classes_tag}_{variant}.pdf"),
            )
    else:
        plot_advantage(
            instance_classes=classes,
            mode=args.mode,
            cache_dir=cache_dir,
            output=Path(f"plot_advantage_{classes_tag}_{args.solver}_{args.mode}.pdf"),
            runtime_key=RUNTIME_KEYS[args.solver],
        )
