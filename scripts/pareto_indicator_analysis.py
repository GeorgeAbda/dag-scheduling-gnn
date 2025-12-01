import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import csv

# Ensure project root import path (so we can import scheduler.* when run as a script)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from scheduler.rl_model.robustness.metrics import (
    pareto_non_dominated,
    hypervolume_2d,
    ideal_nadir,
    normalize_points,
)

Point = Tuple[float, float]


def parse_ckpts(dir_path: Path) -> List[Tuple[int, int, float, float, Path]]:
    """
    Parse *_pareto_*.pt checkpoints under dir_path (non-recursive),
    extracting (iter, step, makespan, active_energy, path).
    """
    pts: List[Tuple[int, int, float, float, Path]] = []
    pat = re.compile(r".*_it(\d+)_st(\d+)\.pt$")
    for fp in sorted(dir_path.glob("*_pareto_*.pt")):
        m = pat.match(fp.name)
        it = int(m.group(1)) if m else -1
        st = int(m.group(2)) if m else -1
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
            pts.append((it, st, mk, ae, fp))
    return pts


def igd(candidate: Sequence[Point], reference_front: Sequence[Point]) -> float:
    if len(reference_front) == 0:
        return float("nan")
    if len(candidate) == 0:
        return float("inf")
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    dists = []
    for r in ref:
        diffs = cand - r[None, :]
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))


def igd_plus(candidate: Sequence[Point], reference_front: Sequence[Point]) -> float:
    if len(reference_front) == 0:
        return float("nan")
    if len(candidate) == 0:
        return float("inf")
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    dists = []
    for r in ref:
        diffs = cand - r[None, :]
        diffs = np.maximum(diffs, 0.0)  # only count positive parts (minimization IGD+)
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))


def spacing(front: Sequence[Point]) -> float:
    """Schott spacing metric over Euclidean distances between nearest neighbors on the front."""
    if len(front) < 2:
        return float("nan")
    pts = np.asarray(front, dtype=float)
    # Compute nearest-neighbor distance for each point
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    d_min = np.min(D, axis=1)
    d_bar = float(np.mean(d_min))
    return float(np.sqrt(np.mean((d_min - d_bar) ** 2)))


def coverage(A: Sequence[Point], B: Sequence[Point]) -> float:
    """Fraction of B dominated by ND(A) (minimization)."""
    if len(B) == 0:
        return 0.0
    A_front = pareto_non_dominated(A)
    cnt = 0
    for p in B:
        dominated = any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for q in A_front)
        if dominated:
            cnt += 1
    return float(cnt / len(B))


def cumulative_fronts(points_sorted: List[Tuple[int, int, float, float]]) -> List[List[Point]]:
    cum: List[List[Point]] = []
    seen: List[Point] = []
    for _it, _st, mk, ae in points_sorted:
        seen.append((mk, ae))
        cum.append(pareto_non_dominated(seen))
    return cum


def _write_csv(path: Path, header: List[str], rows: List[List[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with *_pareto_*.pt checkpoints")
    ap.add_argument("--out", required=True, help="Output directory for CSV and plots")
    ap.add_argument("--normalize", action="store_true", help="Normalize metrics to [0,1] using global ideal/nadir before IGD/spacing")
    ap.add_argument("--ref-front-csv", type=str, default=None, help="Optional path to reference front CSV (columns: mk_ratio,ae_ratio or two columns). If set, IGD/IGD+ use this, and normalization uses points∪ref.")
    args = ap.parse_args()

    d = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = parse_ckpts(d)
    if not ckpts:
        print(f"[warn] No *_pareto_*.pt found under {d}")
        return

    # Sort checkpoints by step, then iter
    ckpts_sorted = sorted(ckpts, key=lambda t: (t[1], t[0]))
    all_pts: List[Point] = [(float(mk), float(ae)) for (_it, _st, mk, ae, _fp) in ckpts_sorted]

    ref_front: List[Point]
    if args.ref_front_csv:
        pts_rf: List[Point] = []
        try:
            with open(args.ref_front_csv, "r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                mk_idx, ae_idx = 0, 1
                if header:
                    try:
                        mk_idx = header.index("mk_ratio")
                        ae_idx = header.index("ae_ratio")
                    except Exception:
                        mk_idx, ae_idx = 0, 1
                for row in r:
                    try:
                        x = float(row[mk_idx])
                        y = float(row[ae_idx])
                        pts_rf.append((x, y))
                    except Exception:
                        continue
        except Exception:
            pts_rf = []
        ref_front = pareto_non_dominated(pts_rf)
    else:
        ref_front = pareto_non_dominated(all_pts)

    comb_for_norm = all_pts + (ref_front if args.ref_front_csv else [])
    ideal, nadir = ideal_nadir(comb_for_norm)
    if args.normalize:
        ref_front_n = normalize_points(ref_front, ideal, nadir)
        hv_ref = (1.05, 1.05)
    else:
        ref_front_n = ref_front
        hv_ref = (nadir[0] * 1.05, nadir[1] * 1.05)

    cum_fronts = cumulative_fronts([(int(it), int(st), float(mk), float(ae)) for (it, st, mk, ae, _fp) in ckpts_sorted])

    hv_vals: List[float] = []
    igd_vals: List[float] = []
    igd_plus_vals: List[float] = []
    spacing_vals: List[float] = []

    for fr in cum_fronts:
        if args.normalize:
            fr_n = normalize_points(fr, ideal, nadir)
            hv = hypervolume_2d(fr_n, hv_ref)
            igd_v = igd(fr_n, ref_front_n)
            igd_p = igd_plus(fr_n, ref_front_n)
            sp = spacing(fr_n)
        else:
            hv = hypervolume_2d(fr, hv_ref)
            igd_v = igd(fr, ref_front)
            igd_p = igd_plus(fr, ref_front)
            # spacing on normalized space for scale invariance
            fr_n = normalize_points(fr, ideal, nadir)
            sp = spacing(fr_n)
        hv_vals.append(hv)
        igd_vals.append(igd_v)
        igd_plus_vals.append(igd_p)
        spacing_vals.append(sp)

    csv_path = out_dir / "metrics_over_time.csv"
    header = ["iter", "step", "makespan", "active_energy", "hv_cum", "igd_cum", "igd_plus_cum", "spacing_cum"]
    rows = []
    for idx, (it, st, mk, ae, _fp) in enumerate(ckpts_sorted):
        rows.append([int(it), int(st), float(mk), float(ae), float(hv_vals[idx]), float(igd_vals[idx]), float(igd_plus_vals[idx]), float(spacing_vals[idx])])
    _write_csv(csv_path, header, rows)

    # Emit final front
    fr_final = cum_fronts[-1]
    final_front_path = out_dir / "final_front.csv"
    fr_sorted = sorted(fr_final)
    _write_csv(final_front_path, ["makespan", "active_energy"], [[float(x), float(y)] for (x, y) in fr_sorted])

    # Coverage of final front over all snapshots (fraction of snapshot points dominated)
    cov = 0.0
    if len(all_pts) > 0:
        A_front = pareto_non_dominated(fr_final)
        cnt = 0
        for p in all_pts:
            if any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for q in A_front):
                cnt += 1
        cov = float(cnt) / float(len(all_pts))
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"final_front_coverage_over_all_snapshots: {cov}\n")

    print(f"[done] Wrote {csv_path}, {final_front_path} and {summary_path}")


if __name__ == "__main__":
    main()
