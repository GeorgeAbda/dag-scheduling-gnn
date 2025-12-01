import numpy as np
from typing import Iterable, Tuple, Sequence, List, Dict

# All metrics below assume a bi-objective minimization setting: (makespan, energy)
# Points are 2D tuples/lists/arrays (M, E). We recommend normalizing to [0,1] using
# ideal and nadir before calling HV/regret if you wish HV to be comparable across instances.

Point = Tuple[float, float]


def pareto_non_dominated(points: Iterable[Point]) -> List[Point]:
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return []
    keep = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        if not keep[i]:
            continue
        p = pts[i]
        # Any point that strictly dominates p will remove it
        dominates = (pts <= p + 1e-12).all(axis=1) & (pts < p - 1e-12).any(axis=1)
        if dominates.any():
            keep[i] = False
            continue
        # Remove points dominated by p
        dominated_by_p = (pts >= p - 1e-12).all(axis=1) & (pts > p + 1e-12).any(axis=1)
        keep &= ~dominated_by_p
        keep[i] = True  # ensure p remains
    return [tuple(pts[i]) for i in range(pts.shape[0]) if keep[i]]


def hypervolume_2d(points: Iterable[Point], reference: Point) -> float:
    """
    Compute 2D hypervolume for a set of points (minimization) with respect to a reference point.
    Assumes reference dominates none of the points and both axes are to be minimized.
    Implementation: sort by objective-1 (M) ascending, sweep areas to the reference.
    """
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(pareto_non_dominated(points), dtype=float)
    if pts.size == 0:
        return 0.0
    # Filter points that are worse than reference (do not contribute positive area)
    mask = (pts[:, 0] <= ref[0] + 1e-12) & (pts[:, 1] <= ref[1] + 1e-12)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0
    # Sort by M ascending, then E ascending
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    hv = 0.0
    prev_m = ref[0]
    prev_e_floor = ref[1]
    # Sweep from worst M (near ref[0]) towards better M
    for i in range(pts.shape[0] - 1, -1, -1):
        m, e = pts[i]
        width = max(0.0, prev_m - m)
        height = max(0.0, prev_e_floor - e)
        hv += width * height
        prev_m = m
        prev_e_floor = min(prev_e_floor, e)
    return float(max(0.0, hv))


def scalarized_value(point: Point, theta: float) -> float:
    """Weighted sum scalarization in [0,1]^2 after normalization."""
    m, e = point
    theta = float(np.clip(theta, 0.0, 1.0))
    return theta * m + (1.0 - theta) * e


def pareto_regret(point: Point, ref_points: Sequence[Point], thetas: Sequence[float]) -> Dict[str, float]:
    """
    Compute regret of a point against a reference set via a grid of scalarizations.
    Returns dict with mean, max, and a small quantile of regrets across theta values.
    """
    if len(ref_points) == 0:
        return {"mean": np.nan, "max": np.nan, "q90": np.nan}
    ref = np.asarray(ref_points, dtype=float)
    vals = []
    for th in thetas:
        s_point = scalarized_value(point, th)
        s_ref = np.min([scalarized_value(tuple(r), th) for r in ref])
        vals.append(s_point - s_ref)
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "q90": float(np.quantile(arr, 0.9)),
    }


def cvar(values: Sequence[float], alpha: float = 0.9) -> float:
    """Conditional Value-at-Risk (average of worst (1-alpha) tail)."""
    if len(values) == 0:
        return float("nan")
    v = np.sort(np.asarray(values, dtype=float))
    k = int(np.floor(alpha * len(v)))
    tail = v[k:]
    if tail.size == 0:
        return float(v[-1])
    return float(np.mean(tail))


def dominance_rate(points: Sequence[Point], ref_points: Sequence[Point]) -> float:
    """
    Fraction of points that are non-dominated by the reference set (i.e., not strictly worse than any ref point).
    1.0 means none of the reference points dominates any of the given points.
    """
    if len(points) == 0:
        return 0.0
    if len(ref_points) == 0:
        return 1.0
    ref = np.asarray(ref_points, dtype=float)
    cnt = 0
    for p in points:
        p_arr = np.asarray(p, dtype=float)
        dominated = ((ref <= p_arr + 1e-12).all(axis=1) & (ref < p_arr - 1e-12).any(axis=1)).any()
        if not dominated:
            cnt += 1
    return float(cnt / len(points))


def normalize_points(points: Sequence[Point], ideal: Point, nadir: Point) -> List[Point]:
    iz = np.asarray(ideal, dtype=float)
    nz = np.asarray(nadir, dtype=float)
    rng = np.maximum(nz - iz, 1e-12)
    out = []
    for m, e in points:
        out.append((float((m - iz[0]) / rng[0]), float((e - iz[1]) / rng[1])))
    return out


def ideal_nadir(points: Sequence[Point]) -> Tuple[Point, Point]:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return (0.0, 0.0), (1.0, 1.0)
    ideal = (float(np.min(pts[:, 0])), float(np.min(pts[:, 1])))
    nadir = (float(np.max(pts[:, 0])), float(np.max(pts[:, 1])))
    return ideal, nadir
