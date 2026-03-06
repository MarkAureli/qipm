#!/usr/bin/env python3
"""Plot quantum advantage curves: fraction of instances where quantum total time < classical runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

GATE_SPEED_RECORD = 5e-11  # seconds — update as needed
N_POINTS = 500             # x-axis resolution

RUNTIME_KEYS = {
    "glpk":      "runtime_glpk",
    "highs-std": "runtime_highs_std",
    "highs-mps": "runtime_highs_mps",
}


def load_data(instance_classes: list[str], cache_dir: Path, runtime_key: str) -> dict[str, list[dict]]:
    """Load .data JSON files for each class. Returns class -> list of data dicts.

    Skips instances missing the requested runtime key or both gate counts.
    """
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
                data = json.loads(data_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rt = data.get(runtime_key)
            if not rt:  # None or 0
                continue
            if data.get("gate_count_qipm1") is None and data.get("gate_count_qipm2") is None:
                continue
            records.append(data)
        if records:
            result[cls] = records
    return result


def crossover_times(gate_counts: np.ndarray, runtimes: np.ndarray) -> np.ndarray:
    """Return runtime / gate_count per instance — the gate time at which quantum breaks even."""
    return runtimes / gate_counts


def advantage_curve(ct: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """For each t in t_values: percentage of instances where t < crossover_time (quantum wins)."""
    return np.array([100.0 * np.mean(t < ct) for t in t_values])


def compute_x_range(all_crossover_times: np.ndarray) -> tuple[float, float]:
    """Compute x-axis range so curve spans ~100% -> ~0%; always includes GATE_SPEED_RECORD."""
    x_min = float(all_crossover_times.min()) / 10
    x_max = float(all_crossover_times.max()) * 10
    x_min = min(x_min, GATE_SPEED_RECORD)
    x_max = max(x_max, GATE_SPEED_RECORD)
    return x_min, x_max


def _extract_gate_counts(records: list[dict], mode: str) -> np.ndarray | None:
    """Extract gate counts from records for the given mode. Returns None if insufficient data."""
    if mode == "qipm1":
        vals = [r["gate_count_qipm1"] for r in records if r.get("gate_count_qipm1") is not None]
    elif mode == "qipm2":
        vals = [r["gate_count_qipm2"] for r in records if r.get("gate_count_qipm2") is not None]
    elif mode == "min":
        vals = []
        for r in records:
            g1 = r.get("gate_count_qipm1")
            g2 = r.get("gate_count_qipm2")
            if g1 is not None and g2 is not None:
                vals.append(min(g1, g2))
            elif g1 is not None:
                vals.append(g1)
            elif g2 is not None:
                vals.append(g2)
    else:
        return None
    return np.array(vals, dtype=np.float64) if vals else None


def plot_advantage(
    instance_classes: list[str],
    mode: str,
    cache_dir: Path,
    output: Path | None,
    runtime_key: str = "runtime_glpk",
) -> None:
    data = load_data(instance_classes, cache_dir, runtime_key)
    if not data:
        print("No data found.")
        return

    # Gather all crossover times to compute x range
    all_cts: list[np.ndarray] = []
    for cls, records in data.items():
        highs = np.array([r[runtime_key] for r in records], dtype=np.float64)
        if mode == "compare":
            for sub in ("qipm1", "qipm2"):
                gc = _extract_gate_counts(records, sub)
                if gc is not None and len(gc) == len(highs):
                    all_cts.append(crossover_times(gc, highs))
        else:
            gc = _extract_gate_counts(records, mode)
            if gc is not None:
                # align highs to same instances (for min mode, some may be filtered)
                if mode == "min":
                    filtered_highs = []
                    for r in records:
                        g1 = r.get("gate_count_qipm1")
                        g2 = r.get("gate_count_qipm2")
                        if g1 is not None or g2 is not None:
                            filtered_highs.append(r[runtime_key])
                    highs_aligned = np.array(filtered_highs, dtype=np.float64)
                else:
                    # filter to those with the required gate count
                    key = "gate_count_" + mode
                    filtered_highs = [r[runtime_key] for r in records if r.get(key) is not None]
                    highs_aligned = np.array(filtered_highs, dtype=np.float64)
                if len(gc) == len(highs_aligned) and len(gc) > 0:
                    all_cts.append(crossover_times(gc, highs_aligned))

    if not all_cts:
        print("No valid data for the requested mode.")
        return

    combined = np.concatenate(all_cts)
    x_min, x_max = compute_x_range(combined)
    t_values = np.geomspace(x_min, x_max, N_POINTS)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    single_class = len(data) == 1

    legend_handles = []

    for i, (cls, records) in enumerate(data.items()):
        color = colors[i % len(colors)]
        highs = np.array([r[runtime_key] for r in records], dtype=np.float64)

        if mode == "compare":
            for sub, ls in (("qipm1", "-"), ("qipm2", "--")):
                key = "gate_count_" + sub
                sub_records = [(r, r[runtime_key]) for r in records if r.get(key) is not None]
                if not sub_records:
                    continue
                gc = np.array([r[key] for r, _ in sub_records], dtype=np.float64)
                hr = np.array([h for _, h in sub_records], dtype=np.float64)
                ct = crossover_times(gc, hr)
                curve = advantage_curve(ct, t_values)
                label = f"{cls} ({sub})" if not single_class else sub
                (line,) = ax.plot(t_values, curve, color=color, linestyle=ls, label=label)
                legend_handles.append(line)
        else:
            # Build aligned (gc, highs) pairs
            if mode == "min":
                filtered = [(r, r[runtime_key]) for r in records
                            if r.get("gate_count_qipm1") is not None or r.get("gate_count_qipm2") is not None]
                gc_list = []
                for r, _ in filtered:
                    g1 = r.get("gate_count_qipm1")
                    g2 = r.get("gate_count_qipm2")
                    gc_list.append(min(v for v in (g1, g2) if v is not None))
                gc = np.array(gc_list, dtype=np.float64)
                hr = np.array([h for _, h in filtered], dtype=np.float64)
            else:
                key = "gate_count_" + mode
                filtered = [(r, r[runtime_key]) for r in records if r.get(key) is not None]
                gc = np.array([r[key] for r, _ in filtered], dtype=np.float64)
                hr = np.array([h for _, h in filtered], dtype=np.float64)

            if len(gc) == 0:
                continue
            ct = crossover_times(gc, hr)
            curve = advantage_curve(ct, t_values)
            label = cls if not single_class else None
            (line,) = ax.plot(t_values, curve, color=color, label=label)
            if not single_class:
                legend_handles.append(line)

    # Vertical line at record gate speed
    vline = ax.axvline(GATE_SPEED_RECORD, color="black", linestyle=":", linewidth=1.2,
                       label=f"Record gate speed ({GATE_SPEED_RECORD:.0e} s)")
    legend_handles.append(vline)

    ax.set_xscale("log")
    ax.set_xlabel("Gate execution time (s)")
    ax.set_ylabel("Instances with quantum advantage (%)")
    ax.set_ylim(-2, 102)

    solver_label = next(k for k, v in RUNTIME_KEYS.items() if v == runtime_key)
    if single_class:
        cls_name = next(iter(data))
        ax.set_title(f"Quantum advantage — {cls_name} ({mode}, vs {solver_label})")
        if mode == "compare":
            h1 = mlines.Line2D([], [], color="C0", linestyle="-", label="qipm1")
            h2 = mlines.Line2D([], [], color="C0", linestyle="--", label="qipm2")
            ax.legend(handles=[h1, h2, vline])
        else:
            ax.legend(handles=[vline])
    else:
        ax.set_title(f"Quantum advantage — all classes ({mode}, vs {solver_label})")
        if mode == "compare":
            # Add line-style legend entries for variants
            h1 = mlines.Line2D([], [], color="gray", linestyle="-", label="qipm1")
            h2 = mlines.Line2D([], [], color="gray", linestyle="--", label="qipm2")
            # class color swatches
            class_handles = [
                mlines.Line2D([], [], color=colors[i % len(colors)], linestyle="-", label=cls)
                for i, cls in enumerate(data)
            ]
            ax.legend(handles=class_handles + [h1, h2, vline], fontsize=8)
        else:
            ax.legend(handles=legend_handles, fontsize=8)

    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot quantum advantage curves from benchmark .data files.",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--mode",
        choices=["qipm1", "qipm2", "compare", "min"],
        default="compare",
        help="Which gate count to use for the advantage curve (default: compare).",
    )
    parser.add_argument(
        "--solver",
        choices=list(RUNTIME_KEYS),
        default="glpk",
        help="Classical solver runtime to compare against (default: glpk).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir if args.cache_dir is not None else Path("cache_dir").resolve()

    if args.instance_classes:
        classes = args.instance_classes
    else:
        classes = [d.name for d in sorted(cache_dir.iterdir()) if d.is_dir()] if cache_dir.is_dir() else []

    plot_advantage(
        instance_classes=classes,
        mode=args.mode,
        cache_dir=cache_dir,
        output=args.output,
        runtime_key=RUNTIME_KEYS[args.solver],
    )
