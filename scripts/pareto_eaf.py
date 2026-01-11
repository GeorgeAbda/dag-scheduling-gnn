import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import csv

# Ensure project root import path
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cogito.gnn_deeprl_model.robustness.metrics import (
    pareto_non_dominated,
    ideal_nadir,
    normalize_points,
)

Point = Tuple[float, float]


PAT_IT_ST = re.compile(r".*_it(\d+)_st(\d+)\.pt$")
PAT_IDX = re.compile(r".*_pareto_(\d+)\.pt$")


def parse_points_from_run(run_dir: Path) -> List[Point]:
    """
    Parse any '*_pareto_*.pt' files under run_dir (non-recursive), extract points, and return ND set.
    Supports both naming patterns with and without it/st.
    """
    pts: List[Point] = []
    files = sorted(run_dir.glob("*_pareto_*.pt"))
    for fp in files:
        mk = float("nan")
        ae = float("nan")
        try:
            obj = torch.load(str(fp), map_location="cpu")
            if isinstance(obj, dict):
                met = obj.get("metrics", {})
                mk = float(met.get("makespan", np.nan))
                ae = float(met.get("active_energy", np.nan))
        except Exception:
            pass
        if np.isfinite(mk) and np.isfinite(ae):
            pts.append((mk, ae))
    return pareto_non_dominated(pts)


def compute_eaf(runs: List[List[Point]], grid_n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical attainment function over a uniform grid in [0,1]^2.
    - runs: list of runs; each run is a list of (m,e) points in normalized space.
    Returns: grid_x, grid_y, alpha matrix of shape (grid_n, grid_n) with values in [0,1].
    """
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    alpha = np.zeros((grid_n, grid_n), dtype=float)
    R = max(1, len(runs))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = np.array([x, y], dtype=float)
            attained = 0
            for run in runs:
                if len(run) == 0:
                    continue
                pts = np.asarray(run, dtype=float)
                # Check if any point dominates z (minimization)
                dom = ((pts[:, 0] <= z[0] + 1e-12) & (pts[:, 1] <= z[1] + 1e-12))
                if np.any(dom):
                    attained += 1
            alpha[i, j] = attained / R
    return xs, ys, alpha


def save_eaf_grid(path: Path, xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "alpha"])
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                w.writerow([float(x), float(y), float(alpha[i, j])])


def extract_level_points(xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray, level: float, tol: float = 0.01) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if abs(float(alpha[i, j]) - level) <= tol:
                pts.append((float(x), float(y)))
    return pts


def write_points_csv(path: Path, pts: List[Point]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])  # normalized space
        for x, y in pts:
            w.writerow([float(x), float(y)])


def main():
    ap = argparse.ArgumentParser(description="Compute EAF grids from multiple runs. Supports pairwise mode (A vs B), N-way via --model, and pseudo EAF via --model-from-checkpoints.")
    # Pairwise (backward-compatible)
    ap.add_argument("--a-name", required=False)
    ap.add_argument("--a-runs", action="append", required=False, help="Run directory for model A; repeat flag for multiple runs")
    ap.add_argument("--b-name", required=False)
    ap.add_argument("--b-runs", action="append", required=False, help="Run directory for model B; repeat flag for multiple runs")
    # N-way: pass multiple models as name:path1,path2; repeat flag for more models
    ap.add_argument("--model", action="append", help="Model spec as name:path1,path2,... (normalized jointly across all models)")
    ap.add_argument("--model-from-checkpoints", action="append", help="Model spec as name:dir; treats each *_pareto_*.pt inside dir as a separate run (singleton point)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--levels", type=str, default="0.5,0.9", help="Comma-separated attainment levels to extract")
    args = ap.parse_args()

    # Parse runs
    def collect_runs(paths: List[str]) -> List[List[Point]]:
        runs: List[List[Point]] = []
        for p in paths:
            pts = parse_points_from_run(Path(p))
            runs.append(pts)
        return runs

    def collect_runs_from_checkpoints(dir_path: str) -> List[List[Point]]:
        """Treat each *_pareto_*.pt file in dir as a separate run with a single point."""
        d = Path(dir_path)
        files = sorted(d.glob("*_pareto_*.pt"))
        runs: List[List[Point]] = []
        for fp in files:
            mk = float("nan"); ae = float("nan")
            try:
                obj = torch.load(str(fp), map_location="cpu")
                if isinstance(obj, dict):
                    met = obj.get("metrics", {})
                    mk = float(met.get("makespan", np.nan))
                    ae = float(met.get("active_energy", np.nan))
            except Exception:
                pass
            if np.isfinite(mk) and np.isfinite(ae):
                runs.append([(float(mk), float(ae))])
        return runs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [float(x.strip()) for x in str(args.levels).split(',') if x.strip()]

    # Branch: N-way if --model is provided; else fallback to pairwise
    if args.model or args.model_from_checkpoints:
        # Parse models
        models: Dict[str, List[str]] = {}
        if args.model:
            for spec in args.model:
                if ':' not in spec:
                    print(f"[warn] Ignoring malformed --model spec (expected name:path1,path2,...): {spec}")
                    continue
                name, paths = spec.split(':', 1)
                path_list = [p.strip() for p in paths.split(',') if p.strip()]
                if not name or not path_list:
                    print(f"[warn] Empty name or paths in --model spec: {spec}")
                    continue
                models[name] = path_list

        models_ckpt: Dict[str, str] = {}
        if args.model_from_checkpoints:
            for spec in args.model_from_checkpoints:
                if ':' not in spec:
                    print(f"[warn] Ignoring malformed --model-from-checkpoints spec (expected name:dir): {spec}")
                    continue
                name, d = spec.split(':', 1)
                if not name or not d.strip():
                    print(f"[warn] Empty name or dir in --model-from-checkpoints spec: {spec}")
                    continue
                models_ckpt[name] = d.strip()

        if not models and not models_ckpt:
            print("[error] No valid --model or --model-from-checkpoints entries parsed")
            return

        # Collect runs per model
        runs_per_model: Dict[str, List[List[Point]]] = {}
        all_pts: List[Point] = []
        for name, paths in models.items():
            runs = collect_runs(paths)
            runs_per_model[name] = runs
            for r in runs:
                all_pts.extend(r)
        for name, d in models_ckpt.items():
            runs = collect_runs_from_checkpoints(d)
            runs_per_model[name] = runs
            for r in runs:
                all_pts.extend(r)

        if len(all_pts) == 0:
            print("[error] No points found across all models")
            return

        ideal, nadir = ideal_nadir(all_pts)

        # Normalize and compute EAF for each model
        xs_ref: np.ndarray = None  # type: ignore
        ys_ref: np.ndarray = None  # type: ignore
        for name, runs in runs_per_model.items():
            runs_n = [normalize_points(r, ideal, nadir) for r in runs]
            xs, ys, alpha = compute_eaf(runs_n, grid_n=int(args.grid))
            if xs_ref is None:
                xs_ref, ys_ref = xs, ys
            save_eaf_grid(out_dir / f"{name}_eaf_grid.csv", xs, ys, alpha)
            # Level points
            for lev in levels:
                pts = extract_level_points(xs, ys, alpha, level=float(lev))
                # 0.5 -> eaf50, 0.9 -> eaf90
                lev_tag = ("%g" % float(lev)).replace('.', '')
                write_points_csv(out_dir / f"{name}_eaf{lev_tag}_points.csv", pts)

        print(f"[done] N-way EAF grids written to {out_dir}")

    else:
        # Pairwise path
        if not args.a_name or not args.b_name or not args.a_runs or not args.b_runs:
            print("[error] Pairwise mode requires --a-name, --b-name, --a-runs, --b-runs")
            return

        A_runs = collect_runs(args.a_runs)
        B_runs = collect_runs(args.b_runs)

        # Normalize jointly across all points from all runs
        all_pts: List[Point] = []
        for r in A_runs + B_runs:
            all_pts.extend(r)
        ideal, nadir = ideal_nadir(all_pts)

        A_runs_n = [normalize_points(r, ideal, nadir) for r in A_runs]
        B_runs_n = [normalize_points(r, ideal, nadir) for r in B_runs]

        # Compute EAF grids
        xs, ys, alpha_A = compute_eaf(A_runs_n, grid_n=int(args.grid))
        _, _, alpha_B = compute_eaf(B_runs_n, grid_n=int(args.grid))

        save_eaf_grid(out_dir / f"{args.a_name}_eaf_grid.csv", xs, ys, alpha_A)
        save_eaf_grid(out_dir / f"{args.b_name}_eaf_grid.csv", xs, ys, alpha_B)

        # Level sets
        for lev in levels:
            a_pts = extract_level_points(xs, ys, alpha_A, level=float(lev))
            b_pts = extract_level_points(xs, ys, alpha_B, level=float(lev))
            lev_tag = ("%g" % float(lev)).replace('.', '')
            write_points_csv(out_dir / f"{args.a_name}_eaf{lev_tag}_points.csv", a_pts)
            write_points_csv(out_dir / f"{args.b_name}_eaf{lev_tag}_points.csv", b_pts)

        # Delta grid (A - B)
        delta = alpha_A - alpha_B
        save_eaf_grid(out_dir / "eaf_delta_grid.csv", xs, ys, delta)

        print(f"[done] EAF grids and level points written to {out_dir}")


if __name__ == "__main__":
    main()
