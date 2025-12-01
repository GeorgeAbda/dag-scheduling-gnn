#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import csv
import re
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional metrics helpers (safe import)
try:
    from scheduler.rl_model.robustness.metrics import (
        pareto_non_dominated,
        ideal_nadir,
        normalize_points,
    )
except Exception:
    pareto_non_dominated = None
    ideal_nadir = None
    normalize_points = None

Point = Tuple[float, float]


def read_points_csv(path: Path) -> List[Point]:
    pts: List[Point] = []
    if not path.exists():
        return pts
    with path.open("r") as f:
        r = csv.reader(f)
        header = next(r, None)
        mk_idx, ae_idx = 0, 1
        if header:
            # Try common header names
            for mk_name, ae_name in (
                ("mk_ratio", "ae_ratio"),
                ("makespan", "active_energy"),
            ):
                try:
                    mk_idx = header.index(mk_name)
                    ae_idx = header.index(ae_name)
                    break
                except Exception:
                    pass
        for row in r:
            try:
                pts.append((float(row[mk_idx]), float(row[ae_idx])))
            except Exception:
                continue
    return pts


def collect_series_for_seed(root: Path, seed: int) -> List[Tuple[str, List[Point]]]:
    series: List[Tuple[str, List[Point]]] = []
    ind_dir = root / f"indicators_seed_{seed}"
    if ind_dir.exists():
        for sub in sorted(ind_dir.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            ff = sub / "final_front.csv"
            pts = read_points_csv(ff)
            if pts:
                series.append((name, pts))
    # NSGA reference
    ref_csv = root / f"nsga_seed_{seed}" / "reference_front.csv"
    nsga_pts = read_points_csv(ref_csv)
    if nsga_pts:
        series.append(("nsga", nsga_pts))
    return series


def plot_series(series: List[Tuple[str, List[Point]]], out_path: Path, normalize: bool, pareto: bool, dpi: int) -> None:
    if plt is None:
        raise SystemExit("matplotlib is not available; cannot plot.")
    if not series:
        return

    # Optional pareto filtering and normalization
    series_proc: List[Tuple[str, List[Point]]] = []
    if pareto and pareto_non_dominated is not None:
        src_for_norm: List[Point] = []
        for name, pts in series:
            nd = pareto_non_dominated(pts)
            series_proc.append((name, nd))
            src_for_norm.extend(nd)
    else:
        series_proc = [(n, p) for (n, p) in series]
        src_for_norm = [pt for (_n, ps) in series_proc for pt in ps]

    if normalize and ideal_nadir is not None and normalize_points is not None and len(src_for_norm) > 0:
        ideal, nadir = ideal_nadir(src_for_norm)
        series_plot = [(n, normalize_points(ps, ideal, nadir)) for (n, ps) in series_proc]
        xlab, ylab = "makespan (normalized)", "active energy (normalized)"
    else:
        series_plot = series_proc
        xlab, ylab = "makespan", "active energy"

    plt.figure(figsize=(6.4, 6.0), dpi=dpi)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # blue, red, green, purple
    for i, (name, pts) in enumerate(series_plot):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        order = np.argsort(xs)
        xs = [xs[k] for k in order]
        ys = [ys[k] for k in order]
        c = colors[i % len(colors)]
        plt.plot(xs, ys, '-', lw=1.5, alpha=0.85, color=c)
        plt.scatter(xs, ys, s=14, alpha=0.9, color=c, label=name)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title("Final Pareto fronts (indicators) + NSGA reference")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Scan an NSGA/indicators root and plot per-seed combined fronts from indicators_seed_* and nsga_seed_*.")
    ap.add_argument("root", type=str, help="Root directory, e.g., figures/config_nsga_indicators_linear20n_v10_h4_w10")
    ap.add_argument("--out-pattern", type=str, default="combined_fronts_seed{seed}.png", help="Filename pattern, supports {seed} placeholder. Written under ROOT unless absolute path is provided.")
    ap.add_argument("--normalize", action="store_true", help="Normalize using union ideal/nadir if helpers available")
    ap.add_argument("--pareto", action="store_true", help="Pareto-filter each series before plotting if helper available")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Find seeds from indicators_seed_* folders
    seeds: List[int] = []
    pat = re.compile(r"^indicators_seed_(\d+)$")
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            seeds.append(int(m.group(1)))

    if not seeds:
        raise SystemExit(f"No indicators_seed_* directories found in {root}")

    for seed in seeds:
        series = collect_series_for_seed(root, seed)
        if not series:
            print(f"[warn] No series found for seed {seed}; skipping")
            continue
        out_name = args.out_pattern.format(seed=seed)
        out_path = (root / out_name) if not out_name.startswith('/') else Path(out_name)
        plot_series(series, out_path, normalize=args.normalize, pareto=args.pareto, dpi=args.dpi)
        print(f"[done] Wrote {out_path}")


if __name__ == "__main__":
    main()
