#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import numpy as np

# Ensure project root on path so we can reuse normalization utilities if needed
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from scheduler.rl_model.robustness.metrics import ideal_nadir, normalize_points

Point = Tuple[float, float]


def compute_eaf(runs: List[List[Point]], grid_n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    alpha = np.zeros((grid_n, grid_n), dtype=float)
    R = max(1, len(runs))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z0, z1 = float(x), float(y)
            attained = 0
            for run in runs:
                if not run:
                    continue
                # any point dominates (minimization, normalized space)
                if any((p[0] <= z0 + 1e-12 and p[1] <= z1 + 1e-12) for p in run):
                    attained += 1
            alpha[i, j] = attained / R
    return xs, ys, alpha


def save_eaf_grid(path: Path, xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "alpha"])
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                w.writerow([float(x), float(y), float(alpha[i, j])])


def load_points_from_csv(csv_path: Path) -> Dict[Tuple[str, str], List[Point]]:
    """Load (makespan, energy_active) points grouped by (agent_train_domain, eval_domain).

    Each row in the CSV corresponds to a single job/seed, so we treat each row
    as a singleton run with one point in objective space.
    """
    groups: Dict[Tuple[str, str], List[Point]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = str(row["agent_train_domain"]).strip()
            d = str(row["eval_domain"]).strip()
            try:
                mk = float(row["makespan"])
                ae = float(row["energy_active"])
            except Exception:
                continue
            key = (a, d)
            groups.setdefault(key, []).append((mk, ae))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute EAF grids from hetero_eval_over_seeds CSV.")
    ap.add_argument("--input", required=True, help="Path to hetero_eval_over_seeds CSV.")
    ap.add_argument("--out-dir", required=True, help="Directory where EAF grid CSVs will be written.")
    ap.add_argument("--grid", type=int, default=100, help="Resolution of the EAF grid per axis (default: 100).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    groups = load_points_from_csv(in_path)
    if not groups:
        raise SystemExit(f"No valid rows found in {in_path}")

    # Jointly normalize across all points from all groups
    all_pts: List[Point] = []
    for pts in groups.values():
        all_pts.extend(pts)
    if not all_pts:
        raise SystemExit("No objective points collected from CSV")

    ideal, nadir = ideal_nadir(all_pts)

    for (agent_train, eval_domain), pts in groups.items():
        pts_n = normalize_points(pts, ideal, nadir)
        # Treat each job as its own run with a single point
        runs = [[p] for p in pts_n]
        xs, ys, alpha = compute_eaf(runs, grid_n=int(args.grid))
        tag = f"{agent_train}_on_{eval_domain}"
        out_path = out_dir / f"{tag}_eaf_grid.csv"
        save_eaf_grid(out_path, xs, ys, alpha)
        print(f"[done] Wrote EAF grid for {tag} to {out_path}")


if __name__ == "__main__":
    main()
