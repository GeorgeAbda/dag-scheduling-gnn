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


def cumulative_fronts(points_sorted: List[Tuple[int, int, float, float]]) -> List[List[Point]]:
    cum: List[List[Point]] = []
    seen: List[Point] = []
    for _it, _st, mk, ae in points_sorted:
        seen.append((mk, ae))
        cum.append(pareto_non_dominated(seen))
    return cum


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
        diffs = np.maximum(diffs, 0.0)  # IGD+ for minimization
        ds = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists.append(float(np.min(ds)))
    return float(np.mean(dists))


def spacing(front: Sequence[Point]) -> float:
    """Schott spacing metric using pure NumPy (no SciPy)."""
    n = len(front)
    if n < 2:
        return float("nan")
    pts = np.asarray(front, dtype=float)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    d_min = np.min(D, axis=1)
    d_bar = float(np.mean(d_min))
    return float(np.sqrt(np.mean((d_min - d_bar) ** 2)))


def write_csv(path: Path, header: List[str], rows: List[List[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Compare two models' Pareto indicators with a shared reference front and normalization.")
    ap.add_argument("--a", required=True, help="Dir A with *_pareto_*.pt (e.g., baseline)")
    ap.add_argument("--b", required=True, help="Dir B with *_pareto_*.pt (e.g., hetero)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epsilon", type=float, default=0.02, help="IGD/IGD+ threshold for time-to-target")
    args = ap.parse_args()

    A_dir = Path(args.a)
    B_dir = Path(args.b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = sorted(parse_ckpts(A_dir), key=lambda t: (t[1], t[0]))
    B = sorted(parse_ckpts(B_dir), key=lambda t: (t[1], t[0]))

    if not A:
        print(f"[error] No checkpoints found in {A_dir}")
        return
    if not B:
        print(f"[error] No checkpoints found in {B_dir}")
        return

    A_pts = [(float(mk), float(ae)) for (_it, _st, mk, ae, _fp) in A]
    B_pts = [(float(mk), float(ae)) for (_it, _st, mk, ae, _fp) in B]

    # Shared reference front and normalization
    all_pts = A_pts + B_pts
    ref_front = pareto_non_dominated(all_pts)
    ideal, nadir = ideal_nadir(all_pts)

    # Normalized space; HV ref just outside [0,1]^2
    hv_ref = (1.05, 1.05)

    A_cum = cumulative_fronts([(it, st, mk, ae) for (it, st, mk, ae, _fp) in A])
    B_cum = cumulative_fronts([(it, st, mk, ae) for (it, st, mk, ae, _fp) in B])

    A_rows: List[List[float]] = []
    B_rows: List[List[float]] = []

    # Pre-normalize reference front once
    ref_front_n = normalize_points(ref_front, ideal, nadir)

    # A
    for idx, (it, st, mk, ae, _fp) in enumerate(A):
        fr = A_cum[idx]
        fr_n = normalize_points(fr, ideal, nadir)
        hv = hypervolume_2d(fr_n, hv_ref)
        igd_v = igd(fr_n, ref_front_n)
        igd_p = igd_plus(fr_n, ref_front_n)
        sp = spacing(fr_n)
        A_rows.append([int(it), int(st), float(mk), float(ae), float(hv), float(igd_v), float(igd_p), float(sp)])

    # B
    for idx, (it, st, mk, ae, _fp) in enumerate(B):
        fr = B_cum[idx]
        fr_n = normalize_points(fr, ideal, nadir)
        hv = hypervolume_2d(fr_n, hv_ref)
        igd_v = igd(fr_n, ref_front_n)
        igd_p = igd_plus(fr_n, ref_front_n)
        sp = spacing(fr_n)
        B_rows.append([int(it), int(st), float(mk), float(ae), float(hv), float(igd_v), float(igd_p), float(sp)])

    write_csv(out_dir / "A_metrics_common.csv", ["iter","step","makespan","active_energy","hv","igd","igd_plus","spacing"], A_rows)
    write_csv(out_dir / "B_metrics_common.csv", ["iter","step","makespan","active_energy","hv","igd","igd_plus","spacing"], B_rows)

    # Summaries
    A_final = A_rows[-1]
    B_final = B_rows[-1]

    def first_step_threshold(rows: List[List[float]], col_idx: int, thr: float) -> int:
        for r in rows:
            if float(r[col_idx]) <= thr:
                return int(r[1])  # step
        return -1

    eps = float(args.epsilon)
    A_t_igd = first_step_threshold(A_rows, 5, eps)
    A_t_igdp = first_step_threshold(A_rows, 6, eps)
    B_t_igd = first_step_threshold(B_rows, 5, eps)
    B_t_igdp = first_step_threshold(B_rows, 6, eps)

    # Coverage of final fronts over union
    def coverage(A_front: Sequence[Point], pts: Sequence[Point]) -> float:
        if len(pts) == 0:
            return 0.0
        A_nd = pareto_non_dominated(A_front)
        cnt = 0
        for p in pts:
            if any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for q in A_nd):
                cnt += 1
        return float(cnt) / float(len(pts))

    A_final_front = A_cum[-1]
    B_final_front = B_cum[-1]
    cov_A_on_all = coverage(A_final_front, all_pts)
    cov_B_on_all = coverage(B_final_front, all_pts)

    summary_path = out_dir / "compare_summary.txt"
    with summary_path.open("w") as f:
        f.write("Model,A,B\n")
        f.write("final_hv,{:.6f},{:.6f}\n".format(A_final[4], B_final[4]))
        f.write("final_igd,{:.6f},{:.6f}\n".format(A_final[5], B_final[5]))
        f.write("final_igd_plus,{:.6f},{:.6f}\n".format(A_final[6], B_final[6]))
        f.write("time_to_igd_leq_{:.3f},{},{}\n".format(eps, A_t_igd, B_t_igd))
        f.write("time_to_igd_plus_leq_{:.3f},{},{}\n".format(eps, A_t_igdp, B_t_igdp))
        f.write("final_front_coverage_over_all,{:.6f},{:.6f}\n".format(cov_A_on_all, cov_B_on_all))

    print(f"[done] Wrote common metrics and summary to {out_dir}")


if __name__ == "__main__":
    main()
