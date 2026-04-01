#!/usr/bin/env python3
"""Plot d·κ histograms: distribution of sparsity × condition number per instance class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def load_dk(
    instance_classes: list[str],
    cache_dir: Path,
    sparsity_key: str,
    cond_key: str,
) -> dict[str, np.ndarray]:
    """Load d·κ products for each class. Returns class -> 1-D array of d*k values."""
    result: dict[str, np.ndarray] = {}
    for cls in instance_classes:
        cls_dir = cache_dir / cls
        if not cls_dir.is_dir():
            continue
        values: list[float] = []
        for instance_dir in sorted(cls_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            data_path = instance_dir / (instance_dir.name + ".data")
            if not data_path.exists():
                continue
            try:
                data = json.loads(data_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            d = data.get(sparsity_key)
            k = data.get(cond_key)
            if d is None or k is None:
                continue
            values.append(float(d) * float(k))
        if values:
            result[cls] = np.array(values, dtype=np.float64)
    return result


def plot_dk_histogram(
    instance_classes: list[str],
    cache_dir: Path,
    sparsity_key: str,
    cond_key: str,
    title: str,
    output: Path,
) -> None:
    """Plot a stacked histogram of d·κ values, one color per instance class."""
    data = load_dk(instance_classes, cache_dir, sparsity_key, cond_key)
    if not data:
        print(f"No data found for {title}; skipping.")
        return

    all_values = np.concatenate(list(data.values()))
    log_min = np.log10(all_values[all_values > 0].min())
    log_max = np.log10(all_values[all_values > 0].max())
    bins = np.logspace(log_min, log_max, N_BINS + 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    classes_sorted = sorted(data.keys(), key=lambda c: float(np.median(data[c])))
    bar_data   = [data[cls] for cls in classes_sorted]
    bar_colors = [CLASS_COLORS.get(cls, "#888888") for cls in classes_sorted]
    bar_labels = [CLASS_LABELS.get(cls, cls) for cls in classes_sorted]

    ax.hist(
        bar_data,
        bins=bins,
        stacked=True,
        color=bar_colors,
        label=bar_labels,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$d \cdot \kappa$", fontsize=11, labelpad=8)
    ax.set_ylabel("number of instances", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, pad=10)

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot d*k histograms from benchmark .data files.",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
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

    plot_dk_histogram(
        instance_classes=classes,
        cache_dir=cache_dir,
        sparsity_key="sparsity_qipm1",
        cond_key="cond_qipm1",
        title="MNES",
        output=Path(f"plot_dk_{classes_tag}_mnes.pdf"),
    )
    plot_dk_histogram(
        instance_classes=classes,
        cache_dir=cache_dir,
        sparsity_key="sparsity_qipm2",
        cond_key="cond_qipm2",
        title="OSS",
        output=Path(f"plot_dk_{classes_tag}_oss.pdf"),
    )
